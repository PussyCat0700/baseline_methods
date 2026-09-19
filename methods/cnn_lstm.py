"""CNN-LSTM

Reference and mechanism: Based on [Agga et al. (2022)](https://doi.org/10.1016/j.epsr.2022.107908), this method combines CNN feature extraction with LSTM temporal modelling.

Task adaptation: Encode historical power/weather and supply forecast weather to a shared 480-step prediction head. Fix the kernel size to 3 and LSTM depth to 1; omit household-consumption inputs.

Pseudocode:
BUILD(c):
    CNN = ConvStack(channels=c.hidden_dim,layers=c.cnn_layers,kernel=3)
    LSTM = LSTM(hidden=c.hidden_dim,layers=1)
    HEAD = Linear(c.hidden_dim) -> ReLU -> Dropout(c.dropout) -> Linear(1)
FORWARD(Wp,Wf,P):
    h = LSTM(CNN(concat(P[...,None],Align15(Wp)))).last_hidden_state
    Q = concat(repeat(h,480),Align15(Wf),forecast_step/480)
    return squeeze_last(HEAD(Q))
TRAIN(T,V,c): return FitNN(BUILD(c),T,V,loss=MSE)
PREDICT(Wp,Wf,P): return Return480(FORWARD(Wp,Wf,P))
"""

import torch
from torch import nn
from common import NeuralMethod, ConvEncoder, ForecastHead, align15

class Network(nn.Module):
    def __init__(self, c, f, features):
        super().__init__()
        if f['conv_stride'] != 1 or f['conv_padding'] != 'same' or f['dropout_location'] != 'prediction_head':
            raise ValueError('CNN-LSTM uses length-preserving convolutions and prediction-head dropout')
        self.cnn = ConvEncoder(features+1,c['hidden_dim'],c['cnn_layers'],f['kernel_size'])
        self.lstm = nn.LSTM(c['hidden_dim'],c['hidden_dim'],num_layers=f['lstm_layers'],batch_first=True)
        self.head = ForecastHead(c['hidden_dim'],features,c['dropout'])

    def forward(self, Wp, Wf, P):
        x = torch.cat([P.unsqueeze(-1),align15(Wp)],-1)
        _, (h, _) = self.lstm(self.cnn(x))
        return self.head(h[-1],Wf)

class CNNLSTM(NeuralMethod):
    def build(self, train):
        return Network(self.config,self.fixed,self.features)
