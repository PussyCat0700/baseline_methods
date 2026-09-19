"""DMOM

Reference and mechanism: Based on [Li et al. (2025)](https://doi.org/10.1109/TSTE.2024.3424932), this method combines power/weather similarity matching with local LSTM modelling.

Task adaptation: Keep a fixed training-window archive and adapt locally using matched historical targets only, without accessing current forecast targets.

Pseudocode:
TRAIN(T,V,c):
    archive = CompletedWindows(T)
    matcher = FitDistanceScales(T,power_terms=[amplitude,fluctuation],weather=True)
    base = FitNN(LSTMSeq2Seq(c.hidden_units,c.dropout),T,V,loss=MSE)
    return archive,matcher,base
PREDICT(Wp,Wf,P):
    a = c.power_similarity_weight
    d = a*PowerDistance(P,archive)+(1-a)*WeatherDistance(Wf,archive)
    matched = archive[SmallestDistances(d,fixed.top_n)]
    local = FineTune(copy(base),matched,budget=fixed.local_budget)
    return Return480(local.eval()(Wp,Wf,P))
"""

import copy
import hashlib
import numpy as np
import torch
from torch import nn
from common import NeuralMethod, pool_time, fit_nn

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        d = c['hidden_units']
        self.encoder = nn.LSTM(features+4,d,f['lstm_layers'],batch_first=True)
        self.decoder = nn.LSTM(features,d,f['lstm_layers'],batch_first=True)
        self.head = nn.Sequential(nn.Dropout(c['dropout']),nn.Linear(d,4))

    def forward(self, Wp, Wf, P):
        _,state = self.encoder(torch.cat([P.reshape(-1,120,4),Wp],-1))
        z,_ = self.decoder(Wf,state)
        return self.head(z).reshape(-1,480)


def descriptors(Wf,P):
    power = np.stack([P.mean(1),P.std(1),np.abs(np.diff(P,axis=1)).mean(1)],1)
    weather = pool_time(Wf,30).reshape(len(P),-1)
    return power,weather

class DMOM(NeuralMethod):
    def build(self, train):return Network(self.config,self.fixed,self.features)

    def _fit(self, train, valid):
        super()._fit(train,valid)
        self.archive = {k:np.array(train[k],copy=True) for k in ('Wp','Wf','P','Y')}
        self.power_desc,self.weather_desc = descriptors(train['Wf'],train['P'])
        self.power_scale = np.maximum(self.power_desc.std(0),1e-6)
        self.weather_scale = np.maximum(self.weather_desc.std(0),1e-6)

    def _predict(self, Wp, Wf, P, time_index):
        pd,wd = descriptors(Wf,P)
        alpha = self.config['power_similarity_weight']
        predictions = []
        for i in range(len(P)):
            distance = alpha*np.mean(((self.power_desc-pd[i])/self.power_scale)**2,1)
            distance += (1-alpha)*np.mean(((self.weather_desc-wd[i])/self.weather_scale)**2,1)
            indices = np.argsort(distance,kind='stable')[:self.fixed['top_n']]
            local = copy.deepcopy(self.net).cpu()
            stable = int.from_bytes(hashlib.sha256(P[i].tobytes()+Wf[i].tobytes()).digest()[:4],'little')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((stable+self.seed) % (2**31))
                opt = torch.optim.AdamW(local.parameters(),lr=self.fixed['local_learning_rate'],weight_decay=self.protocol['training']['weight_decay'])
                local.train()
                a,b = self.protocol['training_steps']
                for _ in range(self.fixed['local_epochs']):
                    for start in range(0,len(indices),self.protocol['training']['batch_size']):
                        idx = indices[start:start+self.protocol['training']['batch_size']]
                        inputs = [torch.from_numpy(self.archive[k][idx]) for k in ('Wp','Wf','P')]
                        y = torch.from_numpy(self.archive['Y'][idx])
                        opt.zero_grad(set_to_none=True)
                        loss = (local(*inputs)[:,a:b]-y[:,a:b]).square().mean()
                        loss.backward()
                        nn.utils.clip_grad_norm_(local.parameters(),self.protocol['training']['gradient_clip'])
                        opt.step()
                local.eval()
                with torch.no_grad():
                    pred = local(torch.from_numpy(Wp[i:i+1]),torch.from_numpy(Wf[i:i+1]),torch.from_numpy(P[i:i+1]))
                    predictions.append(pred.numpy())
        return np.concatenate(predictions)
