"""PVTransNet-E

Reference and mechanism: Based on the encoder variant of [Kim et al. (2024)](https://doi.org/10.1016/j.rser.2024.114479), this method uses Transformer encoding for solar power forecasting.

Task adaptation: Encode forecast weather and predict 480 power values. Each attention block projects to heads×head_dim and back to d_model.

Pseudocode:
BUILD(c):
    EMBED = Linear(c.d_model)
    ENCODER = Transformer(depth=c.layers,heads=c.heads,head_dim=c.head_dim)
    HEAD = Linear(c.d_model) -> Linear(480)
FORWARD(Wp,Wf,P):
    Z = ENCODER(EMBED(Wf)+PositionEncoding)
    return HEAD(flatten(Z))
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

from torch import nn
from common import NeuralMethod, AttentionBlock, position_encoding

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        if f['variant'] != 'E':raise ValueError('This module implements PVTransNet-E')
        d = c['d_model']
        self.embed = nn.Linear(features,d)
        self.register_buffer('position',position_encoding(120,d))
        self.encoder = nn.ModuleList([AttentionBlock(d,c['heads'],c['head_dim'],f['dropout'],f['ff_multiplier']) for _ in range(c['layers'])])
        self.head = nn.Sequential(nn.Flatten(1),nn.Linear(120*d,d),nn.Linear(d,480))

    def forward(self, Wp, Wf, P):
        z = self.embed(Wf)+self.position
        for layer in self.encoder:z = layer(z)
        return self.head(z)

class PVTransNet(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
