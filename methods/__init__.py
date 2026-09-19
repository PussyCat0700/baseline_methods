"""Explicit method registration; each identifier resolves to one independent model."""
import copy
from importlib import import_module

REGISTRY = {
    'heftcom24': ('HEFTCom24', {'estimators','leaves'}),
    'gefcom12': ('GEFCom12', {'estimators','leaves'}),
    'ffnn': ('FFNN', {'hidden_dim','hidden_layers'}),
    'hpe': ('HPE', {'estimators','max_features'}),
    'cnn_rbfnn': ('CNNRBFNN', {'cnn_channels','cnn_blocks'}),
    'ann': ('ANN', {'hidden_dim','layers','dropout'}),
    'crossvivit': ('CrossViViT', {'d_model','encoder_layers','cross_layers','decoder_layers','heads'}),
    'pvtransnet': ('PVTransNet', {'d_model','layers','heads','head_dim'}),
    'cnn_lstm': ('CNNLSTM', {'hidden_dim','cnn_layers','dropout'}),
    'fusionsf': ('FusionSF', {'hidden_dim','heads','decoder_layers'}),
    'hbola': ('HBOLA', {'hidden_size','layers','dropout'}),
    'tft': ('TFT', {'hidden_size','lstm_layers'}),
    'lssvm_rbfnn': ('LSSVMRBFNN', {'C','gamma','rbf_centers'}),
    'dmom': ('DMOM', {'hidden_units','power_similarity_weight','dropout'}),
    'sc_var': ('SCVAR', {'ar_order','sparsity_penalty','correlation_threshold'}),
    'wpd_lstm': ('WPDLSTM', {'hidden_units','dropout'}),
    'atcn': ('ATCN', {'hidden_dim','kernel_size','dropout'}),
    'lstm_gcn_mlp': ('LSTMGCNMLP', {'hidden_units','lstm_layers','dropout'}),
    'narx_ga': ('NARXGA', {'feedback_lags','hidden_units'}),
    'rfs_alo': ('RFsALO', {'trees','leaves'}),
}


def method_class(method_id):
    if method_id not in REGISTRY:
        raise ValueError(f'Unknown method {method_id}; choose from {list(REGISTRY)}')
    return getattr(import_module(f'methods.{method_id}'),REGISTRY[method_id][0])


def create_method(method_id, config, fixed_parameters, protocol):
    cls = method_class(method_id)
    config = copy.deepcopy(config)
    candidate_id = config.pop('id',None)
    if set(config) != REGISTRY[method_id][1]:
        raise ValueError(f'{method_id}: expected fields {sorted(REGISTRY[method_id][1])}, got {sorted(config)}')
    for key,value in config.items():
        if key == 'gamma':
            number = float(value.split('/')[0]) if isinstance(value,str) and value.endswith('/d') else float(value)
            if number <= 0:raise ValueError('gamma must be positive')
        elif key in {'dropout','power_similarity_weight','correlation_threshold'}:
            if not 0 <= value <= 1 or (key=='dropout' and value==1):raise ValueError(f'Invalid {key}')
        elif not isinstance(value,(int,float)) or value <= 0:
            raise ValueError(f'{key} must be positive')
    if protocol['history_hours'] != 120 or protocol['forecast_hours'] != 120 or protocol['power_resolution_minutes'] != 15:
        raise ValueError('The common interface is fixed to 120 hours at 15-min power resolution')
    model = cls(config,fixed_parameters,protocol)
    model.method_id,model.candidate_id = method_id,candidate_id
    return model
