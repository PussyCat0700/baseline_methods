"""ATCN

Reference and mechanism: Based on [Liang and Tang (2022)](https://doi.org/10.1109/TSG.2022.3175451), this method combines causal dilated temporal convolutions with attention.

Task adaptation: Use local power/weather sequences without neighbouring-site inputs, and produce the common power horizon. Retain MAE as the training objective.

Pseudocode:
BUILD(c):
    TCN[k] = CausalDilatedTCN(width=c.hidden_dim,kernel=c.kernel_size)
             for k in 1..fixed.num_bases
    ATTENTION,HEAD = AttentionScorer(),WeatherConditionedHead(out=480)
FORWARD(Wp,Wf,P):
    X = concat(P[...,None],Align15(Wp))
    Z[k] = Dropout(TCN[k](X),c.dropout)
    a = softmax(ATTENTION(Z,Align15(Wf)))
    return HEAD(sum_k(a[k]*Z[k]),Wf)
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MAE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod, MLP, align15

class CausalBlock(nn.Module):
    def __init__(self, input_dim, width, kernel, dilation):
        super().__init__()
        self.left = (kernel-1)*dilation
        self.conv = nn.Conv1d(input_dim,width,kernel,dilation=dilation)
        self.skip = nn.Conv1d(input_dim,width,1) if input_dim != width else nn.Identity()

    def forward(self, x):
        return torch.relu(self.conv(nn.functional.pad(x,(self.left,0)))+self.skip(x))

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        d = c['hidden_dim']
        self.bases = nn.ModuleList()
        for _ in range(f['num_bases']):
            blocks = [CausalBlock(features+1 if k==0 else d,d,c['kernel_size'],dilation) for k,dilation in enumerate(f['dilations'])]
            self.bases.append(nn.Sequential(*blocks,nn.Dropout(c['dropout'])))
        self.query = nn.Linear(features,d)
        self.keys = nn.Linear(d,d)
        self.head = MLP(d+features,d,1,1,c['dropout'])
        self.width = d

    def forward(self, Wp, Wf, P):
        x = torch.cat([P.unsqueeze(-1),align15(Wp)],-1).transpose(1,2)
        z = torch.stack([base(x).mean(-1) for base in self.bases],1)
        weather = align15(Wf)
        weights = torch.einsum('bhd,bkd->bhk',self.query(weather),self.keys(z))/(self.width**.5)
        combined = torch.einsum('bhk,bkd->bhd',weights.softmax(-1),z)
        return self.head(torch.cat([combined,weather],-1)).squeeze(-1)

class ATCN(NeuralMethod):
    def build(self, train):return Network(self.config,self.fixed,self.features)
