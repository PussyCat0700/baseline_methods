"""TFT

Reference and mechanism: Based on [Wang et al. (2024)](https://doi.org/10.1016/j.energy.2024.133577), this method retains variable selection, LSTM encoding/decoding and gated attention.

Task adaptation: Use the shared dynamic inputs without a static-attribute branch, and predict four power values per weather time step.

Pseudocode:
BUILD(c):
    VSNp,VSNf = VariableSelectionNetworks(width=c.hidden_size)
    ENC,DEC = LSTMEncoderDecoder(hidden=c.hidden_size,layers=c.lstm_layers)
    ATTENTION = GatedAttention(heads=7,width=c.hidden_size)
    HEAD = Linear(4)
FORWARD(Wp,Wf,P):
    state = ENC(VSNp(concat(HourlyPower(P),Wp)))
    Z = DEC(VSNf(Wf),initial_state=state)
    return reshape(HEAD(ATTENTION(Z)),[B,480])
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod

class GRN(nn.Module):
    def __init__(self, inputs, width, dropout):
        super().__init__()
        self.skip = nn.Linear(inputs,width) if inputs != width else nn.Identity()
        self.value = nn.Sequential(nn.Linear(inputs,width),nn.ELU(),nn.Linear(width,width),nn.Dropout(dropout))
        self.gate = nn.Linear(width,width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        z = self.value(x)
        return self.norm(self.skip(x)+torch.sigmoid(self.gate(z))*z)

class VariableSelection(nn.Module):
    def __init__(self, variables, width, dropout):
        super().__init__()
        self.transforms = nn.ModuleList([GRN(1,width,dropout) for _ in range(variables)])
        self.weights = nn.Sequential(GRN(variables,width,dropout),nn.Linear(width,variables))

    def forward(self, x):
        values = torch.stack([m(x[...,i:i+1]) for i,m in enumerate(self.transforms)],-2)
        weights = self.weights(x).softmax(-1)
        return (values*weights.unsqueeze(-1)).sum(-2)

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        if f['static_context']:raise ValueError('The shared interface has no static attributes')
        d = c['hidden_size']
        self.past_vsn = VariableSelection(features+4,d,f['dropout'])
        self.future_vsn = VariableSelection(features,d,f['dropout'])
        self.encoder = nn.LSTM(d,d,c['lstm_layers'],batch_first=True)
        self.decoder = nn.LSTM(d,d,c['lstm_layers'],batch_first=True)
        self.attention = nn.MultiheadAttention(d,f['heads'],dropout=f['dropout'],batch_first=True)
        self.post = GRN(d,d,f['dropout'])
        self.gate = nn.Linear(d,d)
        self.head = nn.Linear(d,4)

    def forward(self, Wp, Wf, P):
        history = self.past_vsn(torch.cat([P.reshape(-1,120,4),Wp],-1))
        _, state = self.encoder(history)
        z,_ = self.decoder(self.future_vsn(Wf),state)
        a,_ = self.attention(z,z,z,need_weights=False)
        return self.head(self.post(z+torch.sigmoid(self.gate(a))*a)).reshape(-1,480)

class TFT(NeuralMethod):
    def build(self, train):return Network(self.config,self.fixed,self.features)
