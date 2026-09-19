#!/usr/bin/env python3
"""Tune, train, predict and evaluate the twenty comparison methods."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from common import (BaseMethod, load_config, load_data, check_chronology, scheduled,
                    task_score, calculate_metrics, save_json, config_hash)
from methods import REGISTRY, create_method, method_class

ROOT = Path(__file__).resolve().parent


def read_sites(path, tuning=False):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'site_id','technology','train','validation'}.issubset(rows[0]):
        raise ValueError('Manifest requires site_id,technology,train,validation columns')
    ids = [r['site_id'] for r in rows]
    if len(set(ids)) != len(ids):raise ValueError('Duplicate site IDs')
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',r['site_id']) or r['site_id'] in {'.','..'}:
            raise ValueError('Use simple alphanumeric site IDs')
        if r['technology'] not in {'wind','solar'}:raise ValueError('technology must be wind or solar')
    if tuning:
        counts = {kind:sum(r['technology']==kind for r in rows) for kind in ('wind','solar')}
        if counts != {'wind':30,'solar':30}:
            raise ValueError('Tuning requires the fixed 60-site manifest: 30 wind and 30 solar')
    return rows


def source_hash():
    digest=hashlib.sha256()
    for file in sorted([ROOT/'run.py',ROOT/'common.py',*ROOT.joinpath('methods').glob('*.py')]):
        digest.update(str(file.relative_to(ROOT)).encode());digest.update(file.read_bytes())
    return digest.hexdigest()


def context(args):
    config=load_config(args.config)
    spec=config['methods'][args.method]
    protocol=copy.deepcopy(config['protocol']);protocol['task']=spec['task']
    rows=read_sites(args.sites,tuning=args.command=='tune')
    rows=[r for r in rows if r['technology']==spec['task'].split('_')[0]]
    if not rows:raise ValueError('Manifest has no sites for this task')
    return config,spec,protocol,rows


def read_split(root,row,split,protocol):
    if not row.get(split):raise ValueError(f"{row['site_id']}: missing {split} file")
    path=Path(row[split]);path=path if path.is_absolute() else Path(root)/path
    features=protocol['weather_features'][protocol['task'].split('_')[0]]
    data=load_data(path,features)
    return data if split=='train' else scheduled(data,protocol)


def tune(args):
    config,spec,protocol,rows=context(args)
    out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];candidate_scores={}
    def evaluate(candidate):
        identifier=candidate['id']
        if identifier in candidate_scores:return candidate_scores[identifier]
        site_scores=[]
        for row in rows:
            train=read_split(args.data_root,row,'train',protocol)
            valid=read_split(args.data_root,row,'validation',protocol)
            check_chronology(train,valid)
            seed_scores=[]
            for seed in protocol['seeds']:
                run_protocol=copy.deepcopy(protocol);run_protocol['seed']=seed
                record={'candidate_id':identifier,'site_id':row['site_id'],'seed':seed}
                try:
                    model=create_method(args.method,candidate,spec['fixed'],run_protocol).fit(train,valid)
                    prediction=model.predict(valid['Wp'],valid['Wf'],valid['P'],time_index=valid)
                    score=task_score(prediction,valid['Y'],protocol,model.scaler.power_std)
                    record.update(status='complete',score=score)
                    seed_scores.append(score)
                except Exception as exc:
                    record.update(status='failed',error=str(exc));records.append(record)
                    save_json(out/'site_scores.json',records)
                    raise
                records.append(record)
                save_json(out/'site_scores.json',records)
                print(f"candidate={identifier} site={row['site_id']} seed={seed} score={score:.8g}",flush=True)
            site_scores.append(float(np.mean(seed_scores)))
        candidate_scores[identifier]=float(np.mean(site_scores))
        return candidate_scores[identifier]
    if args.method=='rfs_alo':
        chosen,trace=method_class(args.method).select(evaluate,spec['candidates'],spec['fixed'],protocol['seeds'][0])
        save_json(out/'search_trace.json',trace)
    else:
        for candidate in spec['candidates']:evaluate(candidate)
        chosen=min(spec['candidates'],key=lambda c:(candidate_scores[c['id']],c['id']))
    # All candidates must have scores for all sites and seeds before selection is published.
    expected=3*len(rows)*len(protocol['seeds'])
    if len(records)!=expected or any(r['status']!='complete' for r in records):
        raise ValueError('Incomplete candidate comparison')
    selection={'method_id':args.method,'task':spec['task'],'candidate_id':chosen['id'],
               'configuration':chosen,'fixed':spec['fixed'],'protocol':protocol,
               'config_hash':config_hash(config),'source_hash':source_hash(),
               'site_ids':[r['site_id'] for r in rows],'manifest_sha256':hashlib.sha256(Path(args.sites).read_bytes()).hexdigest(),
               'candidate_scores':candidate_scores}
    save_json(out/'selected.json',selection)
    print(f"Selected configuration {chosen['id']}: {out/'selected.json'}")


def train(args):
    config,spec,protocol,rows=context(args)
    selection=json.loads(Path(args.selection).read_text())
    if selection['method_id']!=args.method or selection['config_hash']!=config_hash(config):
        raise ValueError('Selection does not match the method and frozen configuration file')
    chosen=next((c for c in spec['candidates'] if c['id']==selection['candidate_id']),None)
    if chosen!=selection['configuration'] or selection['fixed']!=spec['fixed'] or selection['protocol']!=protocol:
        raise ValueError('Frozen selection and current settings disagree')
    if selection['source_hash']!=source_hash():
        raise ValueError('Method source changed after tuning; reselect with the current implementation')
    summaries=[]
    for row in rows:
        data={split:read_split(args.data_root,row,split,protocol) for split in ('train','validation','test')}
        check_chronology(data['train'],data['validation'],data['test'])
        for seed in protocol['seeds']:
            run_protocol=copy.deepcopy(protocol);run_protocol['seed']=seed
            model=create_method(args.method,chosen,spec['fixed'],run_protocol).fit(data['train'],data['validation'])
            out=Path(args.save_dir)/row['site_id']/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            model.save(out/'model.pkl')
            test=data['test']
            prediction=model.predict(test['Wp'],test['Wf'],test['P'],time_index=test)
            np.savez_compressed(out/'predictions.npz',Yhat=prediction,issue_times=test['issue_times'],target_times=test['target_times'])
            metrics=calculate_metrics(prediction,test['Y'],protocol)
            record={'site_id':row['site_id'],'seed':seed,'metrics':metrics}
            summaries.append(record)
            save_json(out/'metrics.json',record)
            save_json(out/'protocol.json',protocol)
            save_json(out/'configuration.json',dict(method_id=args.method,configuration=chosen,fixed=spec['fixed'],source_hash=source_hash(),power_mean=model.scaler.power_mean,power_std=model.scaler.power_std))
            print(f"Trained {args.method} site={row['site_id']} seed={seed}",flush=True)
    save_json(Path(args.save_dir)/'site_metrics.json',summaries)
    aggregate={}
    for horizon in summaries[0]['metrics']:
        aggregate[horizon]={}
        for metric in ('MAE','RMSE','R2','Corr'):
            values=[]
            for row in rows:
                seed_values=[s['metrics'][horizon][metric] for s in summaries if s['site_id']==row['site_id']]
                if any(v is None for v in seed_values):
                    values.append(None)
                else:values.append(float(np.mean(seed_values)))
            finite=[v for v in values if v is not None]
            aggregate[horizon][metric]={'mean':float(np.mean(finite)) if finite else None,'valid_sites':len(finite),'total_sites':len(rows)}
    save_json(Path(args.save_dir)/'summary.json',aggregate)


def predict(args):
    model=BaseMethod.load(args.checkpoint)
    data=load_data(args.input,model.features,require_targets=False)
    result=model.predict(data['Wp'],data['Wf'],data['P'],time_index=data)
    out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',Yhat=result,issue_times=data['issue_times'],target_times=data['target_times'])
    save_json(out/'protocol.json',model.protocol)
    print(out/'predictions.npz')


def evaluate(args):
    protocol=json.loads(Path(args.protocol).read_text())
    with np.load(args.predictions,allow_pickle=False) as p, np.load(args.targets,allow_pickle=False) as t:
        from common import time_context,issue_mask
        yhat,truth=p['Yhat'],t['Y']
        if yhat.ndim!=2 or truth.ndim!=2 or yhat.shape[1]!=480 or truth.shape[1]!=480:
            raise ValueError('Predictions and targets must have shape [N,480]')
        pc=time_context(dict(p),len(yhat));tc=time_context(dict(t),len(truth))
        for context in (pc,tc):
            if len(np.unique(context['issue_times']))!=len(context['issue_times']):
                raise ValueError('Duplicate issue times cannot be evaluated')
        pk,tk=issue_mask(pc,protocol),issue_mask(tc,protocol)
        if not pk.any() or not tk.any():raise ValueError('No samples follow the task issue schedule')
        if not np.array_equal(pc['issue_times'][pk],tc['issue_times'][tk]):
            raise ValueError('Predictions must cover exactly the scheduled target issue times')
        metrics=calculate_metrics(yhat[pk],truth[tk],protocol)
    save_json(Path(args.save_dir)/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('tune','train'):
        sub=commands.add_parser(command)
        sub.add_argument('--method',required=True,choices=REGISTRY)
        sub.add_argument('--config',default=str(ROOT/'configs.yaml'))
        sub.add_argument('--data-root',required=True)
        sub.add_argument('--sites',required=True)
        sub.add_argument('--save-dir',required=True)
        if command=='train':sub.add_argument('--selection',required=True)
    sub=commands.add_parser('predict')
    sub.add_argument('--checkpoint',required=True);sub.add_argument('--input',required=True);sub.add_argument('--save-dir',required=True)
    sub=commands.add_parser('evaluate')
    sub.add_argument('--predictions',required=True);sub.add_argument('--targets',required=True)
    sub.add_argument('--protocol',required=True);sub.add_argument('--save-dir',required=True)
    args=parser.parse_args()
    {'tune':tune,'train':train,'predict':predict,'evaluate':evaluate}[args.command](args)

if __name__=='__main__':main()
