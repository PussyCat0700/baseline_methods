"""WPD-LSTM

Reference and mechanism: Based on [Li et al. (2020)](https://doi.org/10.1016/j.apenergy.2019.114216), this method combines wavelet packet decomposition, component LSTMs and linear reconstruction.

Task adaptation: Decompose each input window separately, use future component signals only as training labels, and condition component forecasts on the shared weather inputs.

Pseudocode:
TRAIN(T,V,c):
    for D in [T,V]:
        D.Cp = WindowwiseWPD(D.P,fixed.wavelet,fixed.level,fixed.boundary)
        D.Cy = WindowwiseWPD(D.Y,fixed.wavelet,fixed.level,fixed.boundary)
    for band k:
        Mk = LSTMForecast(hidden=c.hidden_units,output_dropout=c.dropout,out=480)
        Mk = FitNN(Mk,(T.Cp[k],T.Wp,T.Wf,T.Cy[k]),(V.Cp[k],V.Wp,V.Wf,V.Cy[k]),MSE)
    a = FitLinearReconstruction(ComponentPredictions(M,V.inputs),V.Y)
    return M,a
PREDICT(Wp,Wf,P):
    Cp = WindowwiseWPD(P,fixed.wavelet,fixed.level,fixed.boundary)
    return Return480(sum_k(a[k]*Mk(Cp[k],Wp,Wf)))
"""

import numpy as np
import torch
from torch import nn
from common import NeuralMethod, ForecastHead, align15


def reconstructed_bands(values, settings):
    import pywt
    packet = pywt.WaveletPacket(data=np.asarray(values),wavelet=settings['wavelet'],mode=settings['boundary'],maxlevel=settings['level'],axis=-1)
    result = []
    nodes = packet.get_level(settings['level'],order='natural')
    for node in nodes:
        branch = pywt.WaveletPacket(data=None,wavelet=settings['wavelet'],mode=settings['boundary'],maxlevel=settings['level'],axis=-1)
        for other in nodes:
            branch[other.path] = node.data if other.path == node.path else np.zeros_like(other.data)
        result.append(branch.reconstruct(update=False)[...,:values.shape[-1]])
    return np.stack(result,1).astype(np.float32)

class Component(nn.Module):
    def __init__(self, hidden, layers, features, dropout):
        super().__init__()
        self.lstm = nn.LSTM(features+1,hidden,layers,batch_first=True)
        self.head = ForecastHead(hidden,features,dropout)

    def forward(self, band, Wp, Wf):
        _,(h,_) = self.lstm(torch.cat([band.unsqueeze(-1),align15(Wp)],-1))
        return self.head(h[-1],Wf)

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        self.settings = dict(f)
        self.components = nn.ModuleList([Component(c['hidden_units'],f['lstm_layers'],features,c['dropout']) for _ in range(2**f['level'])])
        self.register_buffer('weights',torch.ones(len(self.components)))

    def bands(self,values):
        return torch.as_tensor(reconstructed_bands(values.detach().cpu().numpy(),self.settings),device=values.device)

    def predict_components(self,Wp,Wf,P):
        bands = self.bands(P)
        return torch.stack([m(bands[:,k],Wp,Wf) for k,m in enumerate(self.components)],1)

    def forward(self,Wp,Wf,P):
        return (self.predict_components(Wp,Wf,P)*self.weights[None,:,None]).sum(1)

    def training_loss(self,Wp,Wf,P,Y,steps):
        a,b=steps
        pred = self.predict_components(Wp,Wf,P)
        labels = self.bands(Y)
        return (pred[:,:,a:b]-labels[:,:,a:b]).square().mean()

class WPDLSTM(NeuralMethod):
    def build(self, train):return Network(self.config,self.fixed,self.features)

    def _fit(self, train, valid):
        super()._fit(train,valid)
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(valid['P']),self.protocol['training']['batch_size']):
                inputs=[torch.from_numpy(valid[k][start:start+self.protocol['training']['batch_size']]) for k in ('Wp','Wf','P')]
                predictions.append(self.net.predict_components(*inputs).numpy())
        components=np.concatenate(predictions).transpose(0,2,1).reshape(-1,len(self.net.components))
        weights=np.linalg.lstsq(components,valid['Y'].ravel(),rcond=None)[0]
        self.net.weights.copy_(torch.as_tensor(weights,dtype=torch.float32))
