"""Shared data contract, preprocessing, fitting, scoring and persistence helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

INPUT_KEYS = ('Wp', 'Wf', 'P')
HORIZON = 480


def seed_all(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        result = yaml.safe_load(stream)
    protocol = result['protocol']
    supported = {'selection_score': 'mean_site_standardized_mse',
                 'sm_aggregation': 'equal_mean_short_medium',
                 'tie_break': 'candidate_id_ascending',
                 'scaling': 'per_site_training_zscore', 'output_postprocess': 'none',
                 'timezone': 'Asia/Shanghai'}
    for name, value in supported.items():
        if protocol[name] != value:
            raise ValueError(f'Unsupported {name}: {protocol[name]}')
    if protocol['tuning_sites'] != {'wind': 30, 'solar': 30}:
        raise ValueError('The tuning pool contains 30 wind and 30 solar sites')
    if protocol['weather_features'] != {'wind': 15, 'solar': 12}:
        raise ValueError('Weather feature dimensions must match the shared interface')
    if not protocol['seeds'] or len(set(protocol['seeds'])) != len(protocol['seeds']):
        raise ValueError('Provide a nonempty list of distinct seeds')
    for bounds in [protocol['training_steps'], *protocol['task_windows']['us'], *protocol['task_windows']['sm']]:
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= HORIZON:
            raise ValueError('Invalid output window')
    for name in ('epochs', 'batch_size', 'patience', 'threads'):
        if not isinstance(protocol['training'][name], int) or protocol['training'][name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for key, spec in result['methods'].items():
        rows = spec['candidates']
        if len(rows) != 3 or [r['id'] for r in rows] != [1, 2, 3]:
            raise ValueError(f'{key}: expected exactly three numbered configurations')
        if len({tuple(sorted(r)) for r in rows}) != 1:
            raise ValueError(f'{key}: candidate fields must agree')
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def target_times(issue_times):
    issue = np.asarray(issue_times, dtype='datetime64[m]')
    return issue[:, None] + np.arange(1, 481)[None, :] * np.timedelta64(15, 'm')


def time_context(index, n):
    if index is None:
        return None
    if isinstance(index, dict):
        issue = np.asarray(index['issue_times'], dtype='datetime64[m]')
        targets = np.asarray(index.get('target_times', target_times(issue)), dtype='datetime64[m]')
    else:
        issue = np.asarray(index, dtype='datetime64[m]')
        targets = target_times(issue)
    if issue.shape != (n,) or targets.shape != (n, 480):
        raise ValueError('Invalid issue/target timestamp shapes')
    if np.isnat(issue).any() or not np.array_equal(targets, target_times(issue)):
        raise ValueError('Targets must be issue time + 15, 30, ..., 7200 minutes')
    return {'issue_times': issue, 'target_times': targets}


def check_inputs(Wp, Wf, P, features):
    arrays = [np.asarray(x, dtype=np.float32) for x in (Wp, Wf, P)]
    n = len(arrays[2])
    expected = [(n, 120, features), (n, 120, features), (n, 480)]
    if n == 0:
        raise ValueError('Empty sample set')
    for name, x, shape in zip(INPUT_KEYS, arrays, expected):
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f'{name}: expected finite array of shape {shape}, got {x.shape}')
    return arrays


def check_dataset(data, features, require_targets=True):
    arrays = check_inputs(*(data[k] for k in INPUT_KEYS), features)
    result = dict(zip(INPUT_KEYS, arrays))
    context = time_context(data, len(arrays[2]))
    result.update(context)
    if len(np.unique(result['issue_times'])) != len(result['issue_times']):
        raise ValueError('Duplicate forecast issue times')
    if np.any(np.diff(result['issue_times']) <= np.timedelta64(0, 'm')):
        raise ValueError('Samples must be sorted by issue time')
    if require_targets:
        y = np.asarray(data['Y'], dtype=np.float32)
        if y.shape != arrays[2].shape or not np.isfinite(y).all():
            raise ValueError('Y must be finite with shape [N,480]')
        result['Y'] = y
    return result


def load_data(path, features, require_targets=True):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    return check_dataset(data, features, require_targets)


def subset(data, indices):
    return {k: v[indices] for k, v in data.items()}


def check_chronology(train, valid, test=None):
    datasets = [train, valid] + ([test] if test is not None else [])
    for left, right in zip(datasets, datasets[1:]):
        if left['target_times'].max() >= right['target_times'].min():
            raise ValueError('Target periods overlap across chronological splits')


def issue_mask(data, protocol):
    minutes = (data['issue_times'] - data['issue_times'].astype('datetime64[D]')).astype('timedelta64[m]').astype(int)
    task = protocol['task'].split('_')[-1]
    hours = protocol['issue_hours'][task]
    return np.isin(minutes, np.asarray(hours) * 60)


def scheduled(data, protocol):
    keep = issue_mask(data, protocol)
    if not keep.any():
        raise ValueError('No forecasts follow the configured issue schedule')
    return subset(data, keep)


def window_ranges(protocol):
    return protocol['task_windows'][protocol['task'].split('_')[-1]]


def task_score(pred, truth, protocol, power_std=1.):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 480:
        raise ValueError('Scoring requires matching [N,480] arrays')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Non-finite predictions or targets cannot be scored')
    return float(np.mean([np.mean(((pred[:, a:b] - truth[:, a:b]) / power_std) ** 2)
                          for a, b in window_ranges(protocol)]))


def calculate_metrics(pred, truth, protocol):
    result = {}
    names = ['ultra_short'] if protocol['task'].endswith('_us') else ['short', 'medium']
    for name, (a, b) in zip(names, window_ranges(protocol)):
        y, p = truth[:, a:b].ravel().astype(float), pred[:, a:b].ravel().astype(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Metrics require finite values')
        error = p - y
        variance = np.sum((y-y.mean())**2)
        corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else None
        result[name] = {'n': len(y), 'MAE': float(np.mean(np.abs(error))),
                        'RMSE': float(np.sqrt(np.mean(error**2))),
                        'R2': float(1-np.sum(error**2)/variance) if variance > 0 else None,
                        'Corr': corr}
    return result


class Scaler:
    """One power scale per site and one weather scale per input channel."""
    def fit(self, data):
        hist_times = data['issue_times'][:, None] - np.arange(479, -1, -1)[None, :] * np.timedelta64(15, 'm')
        times = np.concatenate([hist_times.ravel(), data['target_times'].ravel()])
        values = np.concatenate([data['P'].ravel(), data['Y'].ravel()])
        _, indices = np.unique(times, return_index=True)
        power = values[indices].astype(np.float64)
        self.power_mean = float(power.mean())
        self.power_std = max(float(power.std()), 1e-6)
        weather = np.concatenate([data['Wp'], data['Wf']], axis=1).reshape(-1, data['Wp'].shape[-1]).astype(float)
        self.weather_mean = weather.mean(0).astype(np.float32)
        self.weather_std = np.maximum(weather.std(0), 1e-6).astype(np.float32)
        return self

    def transform_inputs(self, Wp, Wf, P):
        return ((Wp-self.weather_mean)/self.weather_std,
                (Wf-self.weather_mean)/self.weather_std,
                (P-self.power_mean)/self.power_std)

    def transform(self, data):
        result = dict(data)
        result.update(zip(INPUT_KEYS, self.transform_inputs(*(data[k] for k in INPUT_KEYS))))
        if 'Y' in data:
            result['Y'] = ((data['Y']-self.power_mean)/self.power_std).astype(np.float32)
        return result

    def inverse_power(self, pred):
        return np.asarray(pred) * self.power_std + self.power_mean


def align15(weather):
    if isinstance(weather, torch.Tensor):
        return weather.repeat_interleave(4, dim=1)
    return np.repeat(weather, 4, axis=1)


def pool_time(x, length):
    n, steps = x.shape[:2]
    if steps % length:
        raise ValueError('Pooling requires an exact time partition')
    shape = (n, length, steps//length) + tuple(x.shape[2:])
    return x.reshape(shape).mean(axis=2)


def tabular_features(Wp, Wf, P):
    n = len(P)
    past = pool_time(P, 120)
    weather = Wp.mean(1)
    horizon = np.broadcast_to(np.arange(1, 481, dtype=np.float32)[None, :, None]/480, (n, 480, 1))
    return np.concatenate([np.broadcast_to(past[:, None, :], (n, 480, 120)),
                           np.broadcast_to(weather[:, None, :], (n, 480, Wp.shape[-1])),
                           align15(Wf), horizon], axis=-1)


def window_features(Wp, Wf, P, weather_pool=30, power_pool=120):
    return np.concatenate([pool_time(Wp, weather_pool).reshape(len(P), -1),
                           pool_time(Wf, weather_pool).reshape(len(P), -1),
                           pool_time(P, power_pool)], axis=1)


class StepRegressor:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, Y):
        self.estimator.fit(X.reshape(-1, X.shape[-1]), Y.reshape(-1))
        return self

    def predict(self, X):
        return self.estimator.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])


class BaseMethod:
    def __init__(self, config, fixed, protocol):
        self.config, self.fixed, self.protocol = copy.deepcopy(config), copy.deepcopy(fixed), copy.deepcopy(protocol)
        self.features = protocol['weather_features'][protocol['task'].split('_')[0]]
        self.seed = int(protocol.get('seed', protocol['seeds'][0]))
        self.fitted = False

    def fit(self, train, validation):
        train = check_dataset(train, self.features)
        validation = scheduled(check_dataset(validation, self.features), self.protocol)
        check_chronology(train, validation)
        seed_all(self.seed, self.protocol['training']['threads'])
        self.scaler = Scaler().fit(train)
        self._fit(self.scaler.transform(train), self.scaler.transform(validation))
        self.fitted = True
        return self

    def predict(self, Wp, Wf, P, *, time_index=None):
        if not self.fitted:
            raise ValueError('Fit or load the model before prediction')
        Wp, Wf, P = check_inputs(Wp, Wf, P, self.features)
        context = time_context(time_index, len(P))
        pred = np.asarray(self._predict(*self.scaler.transform_inputs(Wp, Wf, P), context), dtype=np.float32)
        if pred.shape != (len(P), 480) or not np.isfinite(pred).all():
            raise ValueError('Model must produce finite [N,480] predictions')
        return self.scaler.inverse_power(pred).astype(np.float32)

    def save(self, path):
        if not self.fitted:
            raise ValueError('Cannot save an unfitted model')
        artifact = copy.deepcopy(self)
        if hasattr(artifact, 'net'):
            artifact.net.cpu()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load a trusted model artifact generated by this project."""
        with Path(path).open('rb') as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or not model.fitted:
            raise ValueError('Checkpoint is not a fitted instance of the requested method')
        if hasattr(model, 'net'):
            model.net.cpu().eval()
        return model


def unpack_output(output):
    return output if isinstance(output, tuple) else (output, None)


def network_predict(net, data, batch_size=32):
    device = next(net.parameters()).device
    net.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data['P']), batch_size):
            inputs = [torch.as_tensor(data[k][start:start+batch_size], device=device) for k in INPUT_KEYS]
            pred, _ = unpack_output(net(*inputs))
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs)


def fit_nn(net, train, valid, protocol, loss='mse', chronological=False, epochs=None, learning_rate=None):
    settings = protocol['training']
    net.to(settings['device'])
    device = next(net.parameters()).device
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate or settings['learning_rate'], weight_decay=settings['weight_decay'])
    best, best_score, stale = None, float('inf'), 0
    rng = np.random.default_rng(protocol.get('seed', protocol['seeds'][0]))
    a, b = protocol['training_steps']
    history = []
    for epoch in range(epochs or settings['epochs']):
        net.train()
        order = np.arange(len(train['P'])) if chronological else rng.permutation(len(train['P']))
        for start in range(0, len(order), settings['batch_size']):
            idx = order[start:start+settings['batch_size']]
            inputs = [torch.as_tensor(train[k][idx], device=device) for k in INPUT_KEYS]
            target = torch.as_tensor(train['Y'][idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(net, 'training_loss'):
                objective = net.training_loss(*inputs, target, (a, b))
            else:
                pred, auxiliary = unpack_output(net(*inputs))
                error = pred[:, a:b]-target[:, a:b]
                objective = error.abs().mean() if loss == 'mae' else error.square().mean()
                if auxiliary is not None:
                    objective = objective + auxiliary
            if not torch.isfinite(objective):
                raise ValueError('Non-finite training loss')
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(), settings['gradient_clip'])
            optimizer.step()
            if hasattr(net, 'after_batch'):
                net.after_batch()
        score = task_score(network_predict(net, valid, settings['batch_size']), valid['Y'], protocol)
        history.append({'epoch': epoch+1, 'validation_score': score})
        if score < best_score:
            best_score, stale = score, 0
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    if best is None:
        raise ValueError('Training did not produce a checkpoint')
    net.load_state_dict(best)
    net.cpu().eval()
    return net, history


class NeuralMethod(BaseMethod):
    def _fit(self, train, valid):
        self.net = self.build(train)
        self.net, self.history = fit_nn(self.net, train, valid, self.protocol,
                                       loss=self.fixed.get('loss', 'mse'))

    def _predict(self, Wp, Wf, P, time_index):
        return network_predict(self.net, dict(Wp=Wp, Wf=Wf, P=P), self.protocol['training']['batch_size'])


class MLP(nn.Module):
    def __init__(self, input_dim, width, layers, output_dim, dropout=0.):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = width
        self.layers = nn.Sequential(*blocks, nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.layers(x)


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, width, layers, kernel=3):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [nn.Conv1d(input_dim, width, kernel, padding='same'), nn.ReLU()]
            input_dim = width
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x.transpose(1, 2)).transpose(1, 2)


class ForecastHead(nn.Module):
    def __init__(self, width, features, dropout=0.):
        super().__init__()
        self.head = MLP(width+features+1, width, 1, 1, dropout)

    def forward(self, state, Wf):
        h = state[:, None, :].expand(-1, 480, -1)
        step = torch.arange(1, 481, device=state.device, dtype=state.dtype)[None, :, None]/480
        q = torch.cat([h, align15(Wf), step.expand(len(state), -1, -1)], dim=-1)
        return self.head(q).squeeze(-1)


class ProjectedAttention(nn.Module):
    """Attention with independent model dimension, head count and head width."""
    def __init__(self, dim, heads, head_dim, dropout=0.):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads*head_dim
        self.q, self.k, self.v = nn.Linear(dim, inner), nn.Linear(dim, inner), nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)
        self.dropout = dropout

    def forward(self, query, context=None):
        context = query if context is None else context
        def project(layer, x):
            return layer(x).reshape(len(x), x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = project(self.q, query), project(self.k, context), project(self.v, context)
        z = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
        return self.out(z.transpose(1, 2).reshape(len(query), query.shape[1], -1))


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=.1, ff_multiplier=2):
        super().__init__()
        self.attention = ProjectedAttention(dim, heads, head_dim, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*ff_multiplier), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*ff_multiplier, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, query, context=None):
        z = query + self.drop(self.attention(self.norm1(query), context))
        return z + self.drop(self.ff(self.norm2(z)))


def position_encoding(length, dim):
    positions = torch.arange(length)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.)/dim))
    result = torch.zeros(length, dim)
    result[:, 0::2] = torch.sin(positions*scale)
    result[:, 1::2] = torch.cos(positions*scale[:result[:, 1::2].shape[1]])
    return result
