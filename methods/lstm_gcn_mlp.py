"""LSTM-GCN-MLP

Reference and mechanism: Based on [Yue et al. (2024)](https://doi.org/10.1109/TSTE.2024.3390578), this method combines LSTM temporal encoding, graph convolution and MLP prediction.

Task adaptation: Replace the inter-site graph with a graph of local power/weather variables, constructed from training data only.

Pseudocode:
TRAIN(T,V,c):
    Xnodes = concat(T.P[...,None],Align15(T.Wp))
    A = NormalizeWithSelfLoops(TrainingCorrelationGraph(Xnodes))
    M = NodeLSTM(c.hidden_units,c.lstm_layers) -> Dropout(c.dropout)
        -> GCN(A,depth=fixed.gcn_layers) -> WeatherConditionedMLP(out=480)
    return A,FitNN(M,T,V,loss=MSE)
PREDICT(Wp,Wf,P):
    M.eval()
    Z = M.NodeLSTM(concat(P[...,None],Align15(Wp)))
    return Return480(M.MLP(M.GCN(A,Z),Wf))
"""

import numpy as np
import torch
from torch import nn
from common import NeuralMethod, ForecastHead, align15

class Network(nn.Module):
    def __init__(self, c, f, features, adjacency):
        super().__init__()
        d = c['hidden_units']
        self.lstm = nn.LSTM(1,d,c['lstm_layers'],batch_first=True)
        self.dropout = nn.Dropout(c['dropout'])
        self.graph_layers = nn.ModuleList([nn.Linear(d,d) for _ in range(f['gcn_layers'])])
        self.register_buffer('adjacency',torch.as_tensor(adjacency,dtype=torch.float32))
        self.head = ForecastHead(d,features,c['dropout'])

    def forward(self, Wp, Wf, P):
        x = torch.cat([P.unsqueeze(-1),align15(Wp)],-1)
        batch,steps,nodes = x.shape
        _,(h,_) = self.lstm(x.transpose(1,2).reshape(batch*nodes,steps,1))
        z = self.dropout(h[-1].reshape(batch,nodes,-1))
        for layer in self.graph_layers:
            z = torch.relu(layer(torch.einsum('ij,bjd->bid',self.adjacency,z)))
        return self.head(z.mean(1),Wf)

class LSTMGCNMLP(NeuralMethod):
    def build(self, train):
        x = np.concatenate([train['P'][...,None],align15(train['Wp'])],-1).reshape(-1,self.features+1)
        x = x-x.mean(0)
        covariance = x.T@x
        scale = np.sqrt(np.maximum(np.diag(covariance),1e-12))
        corr = np.abs(covariance/np.outer(scale,scale))
        adjacency = np.where(corr >= self.fixed['graph_threshold'],corr,0)
        np.fill_diagonal(adjacency,self.fixed['kernel_self_loop'])
        degree = 1/np.sqrt(adjacency.sum(1))
        adjacency = degree[:,None]*adjacency*degree[None,:]
        return Network(self.config,self.fixed,self.features,adjacency)
