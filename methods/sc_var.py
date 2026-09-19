"""SC-VAR

Reference and mechanism: Based on [Zhao et al. (2018)](https://doi.org/10.1109/TPWRS.2018.2794450), this adaptation retains correlation-based selection, sparse coefficients and autoregression.

Task adaptation: Replace multiple-site inputs with local power/weather variables, yielding a single-site sparse ARX model; use an L1 penalty as a tractable sparsity approximation.

Pseudocode:
TRAIN(T,V,c):
    Xlag,Y1 = BuildOneStepRows(T,p=c.ar_order,available_weather_only=True)
    J = CorrelationFilter(Xlag,Y1,threshold=c.correlation_threshold)
    J = J union required_power_lags
    beta,bias = argmin MSE(Xlag[...,J]@beta+bias,Y1)+c.sparsity_penalty*L1(beta)
    return beta,bias,J
PREDICT(Wp,Wf,P):
    history,U = copy(P),Align15(Wf)
    for h in 1..480:
        Yhat[h] = LagFeatures(history,U,h)[...,J]@beta+bias
        history.append(Yhat[h])
    return Return480(Yhat)
"""

import numpy as np
from common import BaseMethod, align15

class SCVAR(BaseMethod):
    def _fit(self, train, valid):
        from sklearn.linear_model import Lasso
        p = self.config['ar_order']
        if self.fixed['lags_minutes'] != 15:raise ValueError('AR lags use the common 15-min grid')
        history = np.concatenate([train['P'][:,-p:],train['Y']],1)
        lagged = np.lib.stride_tricks.sliding_window_view(history,p,axis=1)[:,:480,:]
        x = np.concatenate([lagged,align15(train['Wf'])],-1).reshape(-1,p+self.features).astype(float)
        y = train['Y'].ravel().astype(float)
        if len(y)>self.fixed['max_fit_rows']:
            ids = np.linspace(0,len(y)-1,self.fixed['max_fit_rows'],dtype=int); x,y = x[ids],y[ids]
        xc,yc = x-x.mean(0),y-y.mean()
        corr = np.abs(xc.T@yc)/np.maximum(np.sqrt((xc**2).sum(0)*(yc**2).sum()),1e-12)
        self.selected = np.flatnonzero(corr >= self.config['correlation_threshold'])
        self.selected = np.union1d(np.arange(p),self.selected)
        self.regressor = Lasso(alpha=self.config['sparsity_penalty'],max_iter=self.fixed['solver_max_iter'],selection='cyclic')
        self.regressor.fit(x[:,self.selected],y)

    def _predict(self, Wp, Wf, P, time_index):
        p = self.config['ar_order']; weather = align15(Wf)
        history = np.empty((len(P),p+480),dtype=float);history[:,:p]=P[:,-p:]
        for h in range(480):
            x = np.concatenate([history[:,h:h+p],weather[:,h]],1)
            history[:,p+h] = self.regressor.predict(x[:,self.selected])
        return history[:,p:]
