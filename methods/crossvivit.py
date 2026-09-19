"""CrossViViT

Reference and mechanism: Based on [Boussif et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/070a57c5ef1e58cc90201b11d369b3c2-Abstract-Conference.html), this method retains separate encoders and cross-attention between target and context representations.

Task adaptation: Replace satellite context with the shared weather/power sequences. Use explicit QKV projections with head_dim=32 and output dimension c.d_model.

Pseudocode:
BUILD(c):
    FutureEncoder,PastEncoder = Encoders(depth=c.encoder_layers,width=c.d_model)
    CROSS = CrossAttentionStack(depth=c.cross_layers,heads=c.heads,head_dim=32)
    DECODER = TemporalDecoder(depth=c.decoder_layers,width=c.d_model)
    HEAD = Linear(4)
    use explicit QKV projections in all attention blocks
FORWARD(Wp,Wf,P):
    Q = FutureEncoder(Embed(Wf)+FuturePosition)
    M = PastEncoder(Embed(HourlyPower(P),Wp)+PastPosition)
    Z = DECODER(CROSS(Q,M))
    return reshape(HEAD(Z),[B,480])
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod, AttentionBlock, position_encoding

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        d = c['d_model']
        self.future_embed = nn.Linear(features,d)
        self.past_embed = nn.Linear(features+4,d)
        self.register_buffer('position',position_encoding(120,d))
        def block():return AttentionBlock(d,c['heads'],f['head_dim'],f['dropout'],f['ff_multiplier'])
        self.future_encoder = nn.ModuleList([block() for _ in range(c['encoder_layers'])])
        self.past_encoder = nn.ModuleList([block() for _ in range(c['encoder_layers'])])
        self.cross = nn.ModuleList([block() for _ in range(c['cross_layers'])])
        self.decoder = nn.ModuleList([block() for _ in range(c['decoder_layers'])])
        self.head = nn.Linear(d,4)

    def forward(self, Wp, Wf, P):
        q = self.future_embed(Wf)+self.position
        m = self.past_embed(torch.cat([P.reshape(-1,120,4),Wp],-1))+self.position
        for layer in self.future_encoder:q = layer(q)
        for layer in self.past_encoder:m = layer(m)
        for layer in self.cross:q = layer(q,m)
        for layer in self.decoder:q = layer(q)
        return self.head(q).reshape(-1,480)

class CrossViViT(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
