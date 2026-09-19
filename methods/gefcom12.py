"""GEFCom12

Reference and mechanism: Based on [Silva (2014)](https://doi.org/10.1016/j.ijforecast.2013.07.007), this method combines engineered weather features, boosted trees and linear prediction.

Task adaptation: Construct fixed features from the common inputs and timestamps, and predict site-level power over the shared horizon.

Pseudocode:
TRAIN(T,V,c):
    X,Y = EngineeredFeatures(T.inputs,T.timestamps),T.targets
    G = StepRegressor(GBDT(c)).fit(X,Y)
    L = StepRegressor(LinearRegression()).fit(X,Y)
    Xv = EngineeredFeatures(V.inputs,V.timestamps)
    a = argmin_{0<=a<=1} MSE(a*G(Xv)+(1-a)*L(Xv),V.targets)
    return G,L,a
PREDICT(Wp,Wf,P,time_index):
    X = EngineeredFeatures(Wp,Wf,P,time_index)
    return Return480(a*G(X)+(1-a)*L(X))
"""

import numpy as np
from common import BaseMethod, StepRegressor, tabular_features, align15

def engineered_features(Wp,Wf,P,time_index):
    if time_index is None:raise ValueError('GEFCom12 needs issue times for calendar features')
    times = time_index['target_times']
    hour = (times-times.astype('datetime64[D]')).astype('timedelta64[m]').astype(float)/1440
    year_day = (times.astype('datetime64[D]')-times.astype('datetime64[Y]')).astype(int)/365.25
    calendar = np.stack([np.sin(2*np.pi*hour),np.cos(2*np.pi*hour),np.sin(2*np.pi*year_day),np.cos(2*np.pi*year_day)],-1)
    return np.concatenate([tabular_features(Wp,Wf,P),align15(Wf)**2,calendar],-1)

class GEFCom12(BaseMethod):
    def _fit(self, train, valid):
        from lightgbm import LGBMRegressor
        from sklearn.linear_model import LinearRegression
        def features(d):return engineered_features(d['Wp'],d['Wf'],d['P'],d)
        x,xv = features(train),features(valid)
        tree = LGBMRegressor(n_estimators=self.config['estimators'],num_leaves=self.config['leaves'],
            learning_rate=self.fixed['learning_rate'],min_child_samples=self.fixed['min_child_samples'],
            random_state=self.seed,n_jobs=self.fixed['n_jobs'],verbosity=-1)
        self.tree = StepRegressor(tree).fit(x,train['Y'])
        self.linear = StepRegressor(LinearRegression()).fit(x,train['Y'])
        g,l = self.tree.predict(xv),self.linear.predict(xv)
        delta = g-l
        self.weight = float(np.clip(np.sum((valid['Y']-l)*delta)/max(np.sum(delta**2),1e-12),0,1))

    def _predict(self, Wp, Wf, P, time_index):
        x = engineered_features(Wp,Wf,P,time_index)
        return self.weight*self.tree.predict(x)+(1-self.weight)*self.linear.predict(x)
