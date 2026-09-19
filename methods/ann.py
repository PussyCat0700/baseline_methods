"""ANN

Reference and mechanism: Based on [Yadav et al. (2014)](https://doi.org/10.1016/j.rser.2013.12.008), this method combines input selection with an artificial neural network.

Task adaptation: Select features using training data only and change the target from irradiance to power at each forecast step.

Pseudocode:
TRAIN(T,V,c):
    X,Xv = TabularFeatures(T.inputs),TabularFeatures(V.inputs)
    J = SelectFeatures(X,T.targets,rule=fixed.selection_rule)
    M = MLP(width=c.hidden_dim,layers=c.layers,dropout=c.dropout,out=1)
    Dt = (reshape(X[...,J],[-1,len(J)]),flatten(T.targets))
    Dv = (reshape(Xv[...,J],[-1,len(J)]),flatten(V.targets))
    return J,FitNN(M,Dt,Dv,loss=MSE)
PREDICT(Wp,Wf,P):
    X = TabularFeatures(Wp,Wf,P)
    return Return480(reshape(M(X[...,J]),[B,480]))
"""

import numpy as np
import torch
from torch import nn
from common import NeuralMethod, MLP, tabular_features, align15, pool_time

class Network(nn.Module):
    def __init__(self, c, indices):
        super().__init__()
        self.register_buffer('indices',torch.tensor(indices,dtype=torch.long))
        self.mlp = MLP(len(indices),c['hidden_dim'],c['layers'],1,c['dropout'])

    def forward(self, Wp, Wf, P):
        power = pool_time(P,120)[:,None,:].expand(-1,480,-1)
        weather = Wp.mean(1)[:,None,:].expand(-1,480,-1)
        step = torch.arange(1,481,device=P.device,dtype=P.dtype)[None,:,None]/480
        x = torch.cat([power,weather,align15(Wf),step.expand(len(P),-1,-1)],-1)
        return self.mlp(x.index_select(-1,self.indices)).squeeze(-1)

class ANN(NeuralMethod):
    def build(self, train):
        if self.fixed['selection_rule'] != 'absolute_training_correlation':
            raise ValueError('Unknown feature selection rule')
        x = tabular_features(train['Wp'],train['Wf'],train['P']).reshape(-1,121+2*self.features).astype(float)
        y = train['Y'].ravel().astype(float)
        x -= x.mean(0); y -= y.mean()
        score = np.abs(x.T@y)/np.maximum(np.sqrt((x*x).sum(0)*(y*y).sum()),1e-12)
        count = self.fixed['selected_features']
        if not 0 < count <= x.shape[1]:raise ValueError('Invalid selected_features')
        self.selected_features = np.argsort(-score,kind='stable')[:count]
        return Network(self.config,self.selected_features)
