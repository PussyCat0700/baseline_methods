"""FusionSF

Reference and mechanism: Based on [Ma et al. (2024)](https://arxiv.org/abs/2402.05823), this method retains modality encoding, vector quantization and fusion decoding.

Task adaptation: Use the shared power and weather modalities and decode the common power horizon.

Pseudocode:
BUILD(c):
    Ep,Ew = PowerEncoder(c.hidden_dim),WeatherEncoder(c.hidden_dim)
    VQp,VQw = Codebooks(fixed.codebook_size,fixed.quantization_dim)
    DECODER = FusionDecoder(depth=c.decoder_layers,heads=c.heads,output_steps=480)
FORWARD(Wp,Wf,P):
    Zp,Zw = Ep(P),Ew(Wp,Wf)
    Qp,Qw = StraightThroughVQ(Zp,VQp),StraightThroughVQ(Zw,VQw)
    Yhat = DECODER(Qp,Qw)
    Lvq = CodebookLoss + fixed.beta*CommitmentLoss
    return Yhat,Lvq
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE(Yhat,Y)+Lvq)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P).Yhat)
"""

import torch
from torch import nn
from common import NeuralMethod, AttentionBlock, position_encoding

class VectorQuantizer(nn.Module):
    def __init__(self, count, dim, beta):
        super().__init__()
        self.codebook = nn.Embedding(count,dim)
        nn.init.uniform_(self.codebook.weight,-1/count,1/count)
        self.beta = beta

    def forward(self, z):
        distance = (z.unsqueeze(-2)-self.codebook.weight).square().sum(-1)
        q = self.codebook(distance.argmin(-1))
        loss = (q-z.detach()).square().mean()+self.beta*(z-q.detach()).square().mean()
        return z+(q-z).detach(),loss

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        d,q = c['hidden_dim'],f['quantization_dim']
        self.power_encoder = nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,q))
        self.weather_encoder = nn.Sequential(nn.Linear(2*features,d),nn.GELU(),nn.Linear(d,q))
        self.power_vq = VectorQuantizer(f['codebook_size'],q,f['commitment_weight'])
        self.weather_vq = VectorQuantizer(f['codebook_size'],q,f['commitment_weight'])
        self.fuse = nn.Linear(2*q,d)
        self.register_buffer('position',position_encoding(120,d))
        self.decoder = nn.ModuleList([AttentionBlock(d,c['heads'],d//c['heads'],f['dropout']) for _ in range(c['decoder_layers'])])
        self.head = nn.Linear(d,4)

    def forward(self, Wp, Wf, P):
        qp,lp = self.power_vq(self.power_encoder(P.reshape(-1,120,4)))
        qw,lw = self.weather_vq(self.weather_encoder(torch.cat([Wp,Wf],-1)))
        z = self.fuse(torch.cat([qp,qw],-1))+self.position
        for layer in self.decoder:z = layer(z)
        return self.head(z).reshape(-1,480),lp+lw

class FusionSF(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
