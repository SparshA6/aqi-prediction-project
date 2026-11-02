"""Freeze next-day selection/calibration using 2023-2024; never load 2025 here."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/models'))
from forecast_development import feature_table,development_splits
from baselines import make_model
from year_holdout import forecast_windows,normalize_windows
from validation import temporal_validation_split
from seq_models import fit_predict_seq
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def scores(y,p):
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    return dict(n=len(y),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-name',default='2025-ready-v1'); ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--seeds',type=int,default=3); args=ap.parse_args()
    if not args.run_name.replace('-','').replace('_','').isalnum() or args.seeds<1: ap.error('Invalid run name/seeds')
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(2)
    data=ROOT/'data/processed/multiyear_2023_2024'; path=data/'aligned_dataset.csv'
    provenance=json.loads((data/'manifest.json').read_text())
    if provenance['status']!='quality_checked' or sha256(path.read_bytes()).hexdigest()!=provenance['dataset_sha256']: raise ValueError('Unchecked dataset')
    aligned=pd.read_csv(path,parse_dates=['date'])
    if not set(aligned.date.dt.year)<= {2023,2024}: raise ValueError('Development cannot read test-year rows')
    df,features=feature_table(aligned); split=development_splits(df.date); tr=split['train']
    ev={k:v for k,v in split.items() if k in ('selection','calibration')}
    if min(len(tr),*(len(i) for i in ev.values()))<30: raise ValueError('Insufficient split rows')
    out=ROOT/'results/forecast_development/runs'/args.run_name; out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    manifest=dict(status='training',started_utc=datetime.now(timezone.utc).isoformat(),task='next_day',
        protocol='Train Jan 2023-Sep 2024; select Oct 2024; calibrate Nov-Dec 2024; reserve all 2025 for a single locked test',
        dataset_sha256=provenance['dataset_sha256'],device=args.device,seeds=args.seeds,
        rows={k:len(v) for k,v in split.items()},test_year_loaded=False,
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/models').glob('*.py')},
        limitations=['Past AQI is assumed received at forecast issue time.',
        'Delayed weather uses final retrospective ERA5-Land at lag 7/14; publication vintage/Google ingestion time is unverified.',
        'Weather is exploratory; no live availability or 2025 performance claim.',
        'October selection and November-December calibration cover different seasons; interval coverage under shift is not guaranteed.',
        '2024 has already served as an earlier test; new development conclusions require the untouched 2025 test.'])
    def save():
        tmp=out/'manifest.tmp'; tmp.write_text(json.dumps(manifest,indent=2)); tmp.replace(out/'manifest.json')
    save()
    predictions={'Persistence':{k:df.loc[i,'aqi_l1'].to_numpy() for k,i in ev.items()}}
    artifacts={'Persistence':dict(kind='persistence',features=['aqi_l1'])}
    selection_rows=[]
    for family in ('short','long','delayed_weather'):
        for model in ('Ridge','RandomForest','XGBoost'):
            name=f'{model}-{family}'; print(f'Fitting {name}',flush=True)
            estimator=make_model(model,2)
            if model=='XGBoost': estimator.set_params(device=args.device)
            # Predict the change relative to the latest observation, retaining persistence as the anchor.
            estimator.fit(df.loc[tr,features[family]],(df.loc[tr,'y']-df.loc[tr,'aqi_l1']).to_numpy())
            predictions[name]={k:np.clip(df.loc[i,'aqi_l1'].to_numpy()+estimator.predict(df.loc[i,features[family]]),0,500) for k,i in ev.items()}
            artifacts[name]=dict(kind='residual_tabular',model=estimator,features=features[family],anchor='aqi_l1')
            joblib.dump(artifacts[name],out/'models'/f'{name}.joblib')
    raw=forecast_windows(aligned,df,28)
    fit,val=temporal_validation_split(df.loc[tr,'date'].values,28)
    x,mu,sd=normalize_windows(raw[tr[fit]],raw)
    acc={k:[] for k in ev}; checkpoints=[]
    for seed in range(args.seeds):
        print(f'Fitting GRU-28days seed {seed} on {args.device}',flush=True)
        p,epochs,net,(ymu,ysd)=fit_predict_seq('GRU',x[tr],df.loc[tr,'y'].to_numpy(),df.loc[tr,'date'].values,{k:x[i] for k,i in ev.items()},seed,(fit,val),device=args.device)
        weight=f'GRU-28days-s{seed}.pt'; torch.save(net.state_dict(),out/'models'/weight)
        checkpoints.append(dict(weight=weight,target_mean=ymu,target_std=ysd,epochs=epochs))
        for k in ev: acc[k].append(np.clip(p[k],0,500))
    predictions['GRU-28days']={k:np.mean(p,axis=0) for k,p in acc.items()}
    artifacts['GRU-28days']=dict(kind='gru_ensemble',window=28,n_in=x.shape[-1],mu=mu,sd=sd,checkpoints=checkpoints)
    joblib.dump(artifacts['GRU-28days'],out/'models/GRU-28days.joblib')
    for name,p in predictions.items():
        for k,i in ev.items(): selection_rows.append(dict(model=name,split=k,**scores(df.loc[i,'y'].to_numpy(),p[k])))
    metrics=pd.DataFrame(selection_rows); selected=metrics[metrics.split=='selection'].sort_values(['RMSE','model']).iloc[0].model
    idx=ev['calibration']; errors=np.abs(predictions[selected]['calibration']-df.loc[idx,'y'].to_numpy())
    radius=float(np.quantile(errors,min(1.,np.ceil((len(errors)+1)*.9)/len(errors)),method='higher'))
    artifact=artifacts[selected]; artifact.update(selected=selected,interval_radius=radius)
    joblib.dump(artifact,out/'forecast_model.joblib')
    metrics.to_csv(out/'development_metrics.csv',index=False)
    rows=[]
    for k,i in ev.items():
        for name,p in predictions.items():
            r=df.loc[i,['date','station_id','city','y','aqi_l1']].copy(); r['pred']=p[k]; r['model']=name; r['split']=k; rows.append(r)
    pd.concat(rows).to_csv(out/'development_predictions.csv.gz',index=False)
    selected_scores=metrics[(metrics.model==selected)&(metrics.split=='selection')].iloc[0].to_dict()
    manifest.update(status='frozen_awaiting_2025',selected_model=selected,selection_metrics=selected_scores,
        interval_radius=radius,nominal_interval_coverage=.9,
        model_sha256={str(p.relative_to(out)):sha256(p.read_bytes()).hexdigest() for p in (out/'models').iterdir()},
        artifact_sha256=sha256((out/'forecast_model.joblib').read_bytes()).hexdigest(),frozen_utc=datetime.now(timezone.utc).isoformat())
    save(); print(metrics.to_string(index=False),flush=True); print(f'Frozen {selected}; 2025 has not been scored',flush=True)

if __name__=='__main__': main()
