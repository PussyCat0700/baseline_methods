"""HPE

Reference and mechanism: Based on [Vartholomaios et al. (2021)](https://arxiv.org/abs/2107.03825), this method combines Prophet seasonal decomposition with Extra Trees residual prediction.

Task adaptation: Use site timestamps and the shared weather/power features to predict each step of the common horizon.

Pseudocode:
TRAIN(T,V,c):
    S = Prophet.fit(T.timestamps,T.power).seasonal_component
    X = TabularFeatures(T.inputs)
    R = T.targets-S(T.target_times)
    assert c.max_features <= X.feature_count
    E = StepRegressor(ExtraTrees(c.estimators,c.max_features)).fit(X,R)
    return S,E
PREDICT(Wp,Wf,P,time_index):
    R = E(TabularFeatures(Wp,Wf,P))
    return Return480(R+S(time_index.target_times))
"""

import numpy as np
from common import BaseMethod, StepRegressor, tabular_features

class HPE(BaseMethod):
    def seasonal(self,times):
        import pandas as pd
        unique,inverse = np.unique(times.ravel(),return_inverse=True)
        terms = self.prophet.predict(pd.DataFrame({'ds':unique}))['additive_terms'].to_numpy()
        return terms[inverse].reshape(times.shape)

    def _fit(self, train, valid):
        from prophet import Prophet
        from sklearn.ensemble import ExtraTreesRegressor
        import pandas as pd
        historical = train['issue_times'][:,None]-np.arange(479,-1,-1)[None,:]*np.timedelta64(15,'m')
        times = np.concatenate([historical.ravel(),train['target_times'].ravel()])
        values = np.concatenate([train['P'].ravel(),train['Y'].ravel()])
        unique,indices = np.unique(times,return_index=True)
        self.prophet = Prophet(daily_seasonality=self.fixed['daily_seasonality'],weekly_seasonality=self.fixed['weekly_seasonality'],
            yearly_seasonality=self.fixed['yearly_seasonality'],changepoint_prior_scale=self.fixed['changepoint_prior_scale'])
        self.prophet.fit(pd.DataFrame({'ds':unique,'y':values[indices]}),seed=self.seed)
        x = tabular_features(train['Wp'],train['Wf'],train['P'])
        if self.config['max_features'] > x.shape[-1]:raise ValueError('max_features exceeds feature count')
        residual = train['Y']-self.seasonal(train['target_times'])
        self.trees = StepRegressor(ExtraTreesRegressor(n_estimators=self.config['estimators'],max_features=self.config['max_features'],
            random_state=self.seed,n_jobs=self.fixed['n_jobs'])).fit(x,residual)

    def _predict(self, Wp, Wf, P, time_index):
        if time_index is None:raise ValueError('HPE requires issue/target timestamps')
        return self.trees.predict(tabular_features(Wp,Wf,P))+self.seasonal(time_index['target_times'])
