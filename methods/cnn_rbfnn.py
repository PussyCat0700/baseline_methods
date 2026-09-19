"""CNN-RBFNN

Reference and mechanism: Based on [Hong and Rioflorido (2019)](https://doi.org/10.1016/j.apenergy.2019.05.044), this method combines convolutional feature extraction with double-Gaussian radial basis functions.

Task adaptation: Encode local historical power/weather and condition the shared RBF prediction head on forecast weather.

Pseudocode:
BUILD(c):
    CNN = ConvStack(channels=c.cnn_channels,blocks=c.cnn_blocks)
    RBF = DoubleGaussianRBF(centers=fixed.rbf_centers,widths=fixed.width_rule)
    HEAD = Linear(1)
FORWARD(Wp,Wf,P):
    Z = Pool(CNN(concat(P[...,None],Align15(Wp))))
    for h in 1..480:
        Xh = concat(Z,Align15(Wf)[h],h/480)
        Yhat[h] = HEAD(RBF(Xh))
    return stack(Yhat)
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod, ConvEncoder, align15

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        self.cnn = ConvEncoder(features+1,c['cnn_channels'],c['cnn_blocks'],f['kernel_size'])
        self.project = nn.Linear(c['cnn_channels']+features+1,f['rbf_projection_dim'])
        self.centers = nn.Parameter(torch.randn(f['rbf_centers'],f['rbf_projection_dim'])*.2)
        widths = torch.tensor(f['initial_widths']).expand(f['rbf_centers'],-1).clone()
        self.log_width = nn.Parameter(torch.log(torch.expm1(widths)))
        self.mix_logits = nn.Parameter(torch.zeros(f['rbf_centers'],2))
        self.head = nn.Linear(f['rbf_centers'],1)

    def forward(self, Wp, Wf, P):
        z = self.cnn(torch.cat([P.unsqueeze(-1),align15(Wp)],-1)).mean(1)
        h = z[:,None,:].expand(-1,480,-1)
        t = torch.arange(1,481,device=P.device,dtype=P.dtype)[None,:,None]/480
        x = self.project(torch.cat([h,align15(Wf),t.expand(len(P),-1,-1)],-1))
        distance = (x.unsqueeze(-2)-self.centers).square().mean(-1)
        width = nn.functional.softplus(self.log_width)+1e-5
        basis = torch.exp(-distance.unsqueeze(-1)/(2*width.square()))
        basis = (basis*self.mix_logits.softmax(-1)).sum(-1)
        return self.head(basis).squeeze(-1)

class CNNRBFNN(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
