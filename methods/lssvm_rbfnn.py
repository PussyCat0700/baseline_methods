"""LSSVM+RBFNN

Reference and mechanism: Based on [Shi et al. (2014)](https://doi.org/10.1109/TSG.2013.2283269), this method combines LS-SVM and RBFNN predictions using grey relational weighting conditioned on wind regimes.

Task adaptation: Use fixed window features and multi-output prediction. If needed, fix a training-only support-sample budget before tuning.

Pseudocode:
TRAIN(T,V,c):
    X,Y = WindowFeatures(T.inputs),T.targets
    Kij = exp(-c.gamma*norm(Xi-Xj)^2)
    bias,dual = SolveLSSVMBlockSystem(K+I/c.C,Y)
    R = RBFNetwork(centers=c.rbf_centers).fit(X,Y)
    A = FitGreyRelationalWindWeights(V,LS_SVM(X,bias,dual),R)
    return X,bias,dual,R,A
PREDICT(Wp,Wf,P):
    x = WindowFeatures(Wp,Wf,P)
    s,r = Kernel(x,X)@dual+bias,R(x)
    return Return480(A(Wf)*s+(1-A(Wf))*r)
"""

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from common import BaseMethod, window_features, align15

class LSSVMRBFNN(BaseMethod):
    def features_for(self,Wp,Wf,P):
        return window_features(Wp,Wf,P,self.fixed['weather_pool'],self.fixed['power_pool'])

    def kernel(self,x,z):return np.exp(-self.gamma*cdist(x,z,'sqeuclidean'))

    def rbf(self,x):
        distance = cdist(x,self.centers,'sqeuclidean')/x.shape[1]
        return np.exp(-distance/(2*self.fixed['rbf_width']**2))

    def components(self,x):
        return self.kernel(x,self.support)@self.dual+self.bias,self.rbf(x)@self.rbf_weights

    def grey_weights(self,truth,ls,rbf):
        error = np.stack([np.abs(ls-truth),np.abs(rbf-truth)],-1)
        maximum = max(float(error.max()),1e-12)
        coefficient = (float(error.min())+self.fixed['grey_rho']*maximum)/(error+self.fixed['grey_rho']*maximum)
        grades = coefficient.mean(0)
        return grades/grades.sum()

    def _fit(self, train, valid):
        x = self.features_for(train['Wp'],train['Wf'],train['P']).astype(float)
        y = train['Y'].astype(float)
        expression = self.config['gamma']
        self.gamma = float(expression.split('/')[0])/x.shape[1] if isinstance(expression,str) and expression.endswith('/d') else float(expression)
        rng = np.random.default_rng(self.seed)
        ids = np.sort(rng.choice(len(x),min(len(x),self.fixed['support_budget']),replace=False))
        self.support = x[ids]
        n = len(ids)
        matrix = np.empty((n+1,n+1));matrix[0,0]=0;matrix[0,1:]=1;matrix[1:,0]=1
        matrix[1:,1:] = self.kernel(self.support,self.support)+np.eye(n)/self.config['C']
        solution = solve(matrix,np.concatenate([np.zeros((1,480)),y[ids]],0),assume_a='sym')
        self.bias,self.dual = solution[0],solution[1:]
        count = self.config['rbf_centers']
        self.centers = x[rng.choice(len(x),count,replace=len(x)<count)].copy()
        phi = self.rbf(x)
        self.rbf_weights = solve(phi.T@phi+np.eye(count)*self.fixed['rbf_ridge'],phi.T@y,assume_a='pos')
        xv = self.features_for(valid['Wp'],valid['Wf'],valid['P'])
        ls,rbf = self.components(xv)
        index = self.fixed['wind_regime_feature']
        wind_train = align15(train['Wf'])[...,index]
        bins = self.fixed['regime_bins']
        self.boundaries = np.quantile(wind_train,np.arange(1,bins)/bins)
        regimes = np.digitize(align15(valid['Wf'])[...,index],self.boundaries)
        fallback = self.grey_weights(valid['Y'].ravel(),ls.ravel(),rbf.ravel())
        self.weights = np.stack([self.grey_weights(valid['Y'][regimes==k],ls[regimes==k],rbf[regimes==k])
                                 if np.any(regimes==k) else fallback for k in range(bins)])

    def _predict(self, Wp, Wf, P, time_index):
        ls,rbf = self.components(self.features_for(Wp,Wf,P))
        regimes = np.digitize(align15(Wf)[...,self.fixed['wind_regime_feature']],self.boundaries)
        weights = self.weights[regimes]
        return weights[...,0]*ls+weights[...,1]*rbf
