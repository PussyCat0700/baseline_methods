"""FFNN

Reference and mechanism: Based on the FFNN component of [Bhaskar and Singh (2012)](https://doi.org/10.1109/TSTE.2011.2182215), this method learns a nonlinear feed-forward mapping to power.

Task adaptation: Feed available weather forecasts directly to the network, omit the AWNN front end, and predict the common 480-step horizon.

Pseudocode:
BUILD(c):
    MLP = c.hidden_layers blocks of Linear(c.hidden_dim),ReLU,Dropout(fixed.dropout)
    HEAD = Linear(480)
FORWARD(Wp,Wf,P):
    X = concat(flatten(PoolTime(Wf,30)),PoolTime(P,120))
    return HEAD(MLP(X))
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod, MLP, pool_time

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        self.weather_pool, self.power_pool = f['weather_pool'], f['power_pool']
        self.mlp = MLP(features*self.weather_pool+self.power_pool, c['hidden_dim'], c['hidden_layers'], 480, f['dropout'])

    def forward(self, Wp, Wf, P):
        x = torch.cat([pool_time(Wf,self.weather_pool).flatten(1),pool_time(P,self.power_pool)],1)
        return self.mlp(x)

class FFNN(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
