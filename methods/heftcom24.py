"""HEFTCom24

Reference and mechanism: Based on [Pu et al., HEFTCom2024](https://arxiv.org/abs/2505.10367), this method combines multiple forecasting models through stacking.

Task adaptation: Use fixed feature views from the shared weather and power inputs for site-level point forecasting; omit the trading component.

Pseudocode:
TRAIN(T,V,c):
    X,Y = TabularFeatures(T.inputs),T.targets
    for feature_view b in fixed.views:
        OOF[b] = ForwardTimeOOF(GBDT(c),X[...,b],Y)
        G[b] = StepRegressor(GBDT(c)).fit(X[...,b],Y)
    S = LinearCombiner.fit(common_valid_rows(OOF),corresponding_targets)
    return G,S
PREDICT(Wp,Wf,P):
    X = TabularFeatures(Wp,Wf,P)
    return Return480(S(concat_b(G[b](X[...,b]))))
"""

import numpy as np
from common import BaseMethod, StepRegressor, tabular_features

class HEFTCom24(BaseMethod):
    def _estimator(self):
        from lightgbm import LGBMRegressor
        return StepRegressor(LGBMRegressor(n_estimators=self.config['estimators'],num_leaves=self.config['leaves'],
            learning_rate=self.fixed['learning_rate'],min_child_samples=self.fixed['min_child_samples'],
            random_state=self.seed,n_jobs=self.fixed['n_jobs'],verbosity=-1))

    def _views(self, dimension):
        mapping = {'all':np.arange(dimension),'power_weather':np.arange(dimension-1),
                   'weather_horizon':np.arange(120,dimension)}
        return [mapping[name] for name in self.fixed['feature_views']]

    def _fit(self, train, valid):
        from sklearn.linear_model import LinearRegression
        x = tabular_features(train['Wp'],train['Wf'],train['P'])
        y = train['Y']; self.views = self._views(x.shape[-1])
        folds = self.fixed['oof_folds']
        if len(y) < folds+1:raise ValueError('Insufficient training windows for forward stacking')
        bounds = np.linspace(0,len(y),folds+2,dtype=int)
        oof = np.full((len(y),480,len(self.views)),np.nan)
        self.oof_indices = []
        for j in range(1,folds+1):
            validation = np.arange(bounds[j],bounds[j+1])
            cutoff = train['target_times'][validation].min()
            fit_indices = np.flatnonzero(train['target_times'].max(1) < cutoff)
            if not len(fit_indices):continue
            self.oof_indices.append({'train':fit_indices.tolist(),'validation':validation.tolist()})
            for v,cols in enumerate(self.views):
                model = self._estimator().fit(x[fit_indices][...,cols],y[fit_indices])
                oof[validation,:,v] = model.predict(x[validation][...,cols])
        keep = np.isfinite(oof).all((1,2))
        if not keep.any():raise ValueError('No leakage-free forward fold: supply a longer training period')
        self.stacker = LinearRegression().fit(oof[keep].reshape(-1,len(self.views)),y[keep].ravel())
        self.models = [self._estimator().fit(x[...,cols],y) for cols in self.views]

    def _predict(self, Wp, Wf, P, time_index):
        x = tabular_features(Wp,Wf,P)
        base = np.stack([m.predict(x[...,cols]) for m,cols in zip(self.models,self.views)],-1)
        return self.stacker.predict(base.reshape(-1,len(self.models))).reshape(len(P),480)
