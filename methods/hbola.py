"""HBOLA

Reference and mechanism: Based on [Pan et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3304898), this method combines layer-wise LSTM predictions with Hedge weighting and drift adaptation.

Task adaptation: Perform chronological updates on training data, then freeze the network and Hedge weights for final evaluation.

Pseudocode:
TRAIN(T,V,c):
    M = LayerwiseLSTM(hidden=c.hidden_size,depth=c.layers,dropout=c.dropout,out=480)
    a = UniformWeights(c.layers)
    for X,Y in ChronologicalBatches(T):
        pred[k] = M.layer_head(k,X)
        loss[k] = MSE(pred[k],Y)
        GradientUpdate(M,sum_k(a[k]*loss[k]))
        a[k] = max(fixed.epsilon,a[k]*exp(-fixed.eta*loss[k])); normalize(a)
        ApplyFixedDriftRule(M,a,max_depth=c.layers)
    return ValidationSelectedCheckpoint(M,a,V)
PREDICT(Wp,Wf,P):
    return Return480(sum_k(a[k]*M.layer_head(k,Wp,Wf,P)))
"""

import torch
from torch import nn
from common import NeuralMethod, ForecastHead, fit_nn

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        d,n = c['hidden_size'],c['layers']
        self.layers = nn.ModuleList([nn.LSTM(features+4 if k == 0 else d,d,batch_first=True) for k in range(n)])
        self.heads = nn.ModuleList([ForecastHead(d,features,c['dropout']) for _ in range(n)])
        self.register_buffer('weights',torch.full((n,),1/n))
        self.settings,self.recent = dict(f),[]

    def predictions(self, Wp, Wf, P):
        z = torch.cat([P.reshape(-1,120,4),Wp],-1)
        result = []
        for layer,head in zip(self.layers,self.heads):
            z,(h,_) = layer(z)
            result.append(head(h[-1],Wf))
        return torch.stack(result,0)

    def forward(self, Wp, Wf, P):
        return (self.predictions(Wp,Wf,P)*self.weights[:,None,None]).sum(0)

    def training_loss(self, Wp, Wf, P, Y, steps):
        a,b = steps
        loss = (self.predictions(Wp,Wf,P)[...,a:b]-Y[None,:,a:b]).square().mean((1,2))
        self.pending_loss = loss.detach()
        return (self.weights*loss).sum()

    def after_batch(self):
        with torch.no_grad():
            f = self.settings
            updated = self.weights*torch.exp(-f['hedge_learning_rate']*self.pending_loss.clamp(max=100))
            updated = updated.clamp(min=f['weight_floor'])
            self.weights.copy_(updated/updated.sum())
            self.recent.append(float(self.pending_loss.mean()))
            window = f['drift_window']
            self.recent = self.recent[-2*window:]
            if len(self.recent) == 2*window:
                old = sum(self.recent[:window])/window
                new = sum(self.recent[window:])/window
                if new > f['drift_factor']*max(old,1e-8):
                    self.weights.fill_(1/len(self.weights)); self.recent.clear()

class HBOLA(NeuralMethod):
    def build(self, train):return Network(self.config,self.fixed,self.features)

    def _fit(self, train, valid):
        self.net,self.history = fit_nn(self.build(train),train,valid,self.protocol,chronological=True)
