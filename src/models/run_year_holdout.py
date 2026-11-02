"""Fit/select/calibrate exclusively in 2023, then evaluate the untouched 2024 year."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'src/models'))
from year_holdout import split_years, forecast_windows, normalize_windows, paired_week_bootstrap

def save(out, manifest):
    temp=out/'manifest.tmp'; temp.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); temp.replace(out/'manifest.json')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-name',default='2023-to-2024-v1')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--seeds',type=int,default=3)
    args=ap.parse_args()
    if not args.run_name.replace('-','').replace('_','').isalnum() or args.seeds<1: ap.error('Invalid run name or seed count')
    data=ROOT/'data/processed/multiyear_2023_2024'
    provenance=json.loads((data/'manifest.json').read_text())
    aligned_path=data/'aligned_dataset.csv'
    if provenance['status']!='quality_checked' or sha256(aligned_path.read_bytes()).hexdigest()!=provenance['dataset_sha256']: raise ValueError('Multi-year data are unchecked or changed')
    os.environ['AQI_ALIGNED_CSV']=str(aligned_path)
    os.environ['AQI_GEE_DIR']=str(data/'gee')
    from data import load_dataset, FEATURE_SETS
    from baselines import make_model
    from run_forecast import forecast_table, FEATURES, metrics
    from seq_data import build_windows, Preprocessor
    from seq_models import fit_predict_seq
    from validation import temporal_validation_split
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(2)
    out=ROOT/'results/year_holdout/runs'/args.run_name
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  protocol='Fit Jan-Sep 2023; select Oct 2023; calibrate Nov-Dec 2023; test all available 2024 station-days',
                  frozen_before_test_evaluation=True, dataset_sha256=provenance['dataset_sha256'], training_device=args.device,
                  seeds=args.seeds, packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','xgboost','scikit-learn']},
                  source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/models').glob('*.py')},
                  limitations=['2024 daily one-step forecasts use observations received on previous 2024 days, without fitting on 2024 targets.',
                               'Same-day estimation uses retrospective same-day weather/satellite observations; it is not a forecast.',
                               'Only observed eligible station-days are scored; missingness and station changes can bias comparisons.',
                               'Interval calibration on 2023 is not guaranteed to cover 90% under year-to-year shift.'])
    save(out,manifest)
    aligned=pd.read_csv(aligned_path,parse_dates=['date'])
    same=load_dataset()
    same=same[same.aqi_l1.notna()].reset_index(drop=True)
    forecast=forecast_table(aligned)
    all_metrics=[]; all_group=[]; all_predictions=[]; seeds_metrics=[]
    for mode,df in [('same_day',same),('next_day',forecast)]:
        print(f'Preparing {mode}: {len(df)} rows',flush=True)
        splits=split_years(df.date)
        if any(len(x)<30 for x in splits.values()): raise ValueError('Insufficient split rows')
        tr=splits['train']; ev={k:v for k,v in splits.items() if k!='train'}
        manifest.setdefault('rows',{})[mode]={k:len(v) for k,v in splits.items()}; save(out,manifest)
        reference='aqi_l1' if mode=='same_day' else 'latest_aqi'
        predictions={'Persistence':{k:df.loc[idx,reference].to_numpy() for k,idx in ev.items()}}
        artifacts={'Persistence':dict(kind='persistence',feature=reference)}
        env=FEATURE_SETS['SW+G+T']; hist=FEATURE_SETS['SW+G+T+H']
        configs=[('Ridge-env','Ridge',env),('Ridge-history','Ridge',hist),('XGBoost-env','XGBoost',env),('XGBoost-history','XGBoost',hist),('RandomForest-history','RandomForest',hist)] if mode=='same_day' else [(m+'-history',m,FEATURES) for m in ['Ridge','RandomForest','XGBoost']]
        for name, model, cols in configs:
            print(f'Fitting {mode} / {name} using 2023 only',flush=True)
            estimator=make_model(model,2)
            if model=='XGBoost': estimator.set_params(device=args.device)
            estimator.fit(df.loc[tr,cols],df.loc[tr,'y'])
            predictions[name]={k:estimator.predict(df.loc[idx,cols]) for k,idx in ev.items()}
            artifacts[name]=dict(kind='tabular',model=estimator,features=cols)
            joblib.dump(artifacts[name],out/'models'/f'{mode}__{name}.joblib')
        window=7
        fit_mask,val_mask=temporal_validation_split(df.loc[tr,'date'].values,window)
        if mode=='same_day':
            raw=build_windows(df)
            pre=Preprocessor(True).fit(raw[tr[fit_mask]])
            xtr=pre.transform(raw[tr],window)
            xev={k:pre.transform(raw[idx],window) for k,idx in ev.items()}
            norm=dict(mu=pre.mu,sd=pre.sd,cols=pre.cols)
            name='GRU-env'
        else:
            raw=forecast_windows(aligned,df,window)
            transformed,mu,sd=normalize_windows(raw[tr[fit_mask]],raw)
            xtr=transformed[tr]; xev={k:transformed[idx] for k,idx in ev.items()}
            norm=dict(mu=mu,sd=sd,features=['past_aqi','observed_indicator','doy_sin','doy_cos','lat','lon'])
            name='GRU-history'
        acc={k:[] for k in ev}
        for seed in range(args.seeds):
            print(f'Fitting {mode} / {name}, seed {seed}, device {args.device}',flush=True)
            pred,epochs,net,(ymu,ysd)=fit_predict_seq('GRU',xtr,df.loc[tr,'y'].to_numpy(),df.loc[tr,'date'].values,xev,seed,(fit_mask,val_mask),device=args.device)
            torch.save(net.state_dict(),out/'models'/f'{mode}__{name}__s{seed}.pt')
            joblib.dump(dict(normalization=norm,target_mean=ymu,target_std=ysd,window=window,n_in=xtr.shape[-1],epochs=epochs),out/'models'/f'{mode}__{name}__s{seed}.joblib')
            for k in ev: acc[k].append(pred[k])
            seeds_metrics.append(dict(mode=mode,seed=seed,model=name,epochs=epochs,**metrics(df.loc[ev['selection'],'y'].to_numpy(),pred['selection'])))
        predictions[name]={k:np.mean(v,axis=0) for k,v in acc.items()}
        artifacts[name]=dict(kind='gru_ensemble',seeds=args.seeds)
        # Selection is fixed before any test metric is computed; persistence is eligible.
        selection={m:metrics(df.loc[ev['selection'],'y'].to_numpy(),p['selection'])['RMSE'] for m,p in predictions.items()}
        selected=min(selection,key=selection.get)
        joblib.dump(artifacts[selected],out/'models'/f'{mode}__selected.joblib')
        manifest.setdefault('modes',{})[mode]=dict(selected_model=selected,selection_rmse=selection,reference='Persistence')
        save(out,manifest)
        cal=ev['calibration']; residual=np.abs(predictions[selected]['calibration']-df.loc[cal,'y'].to_numpy())
        radius=float(np.quantile(residual,min(1.,np.ceil((len(residual)+1)*.9)/len(residual)),method='higher'))
        for model,preds in predictions.items():
            for split,idx in ev.items():
                all_metrics.append(dict(mode=mode,model=model,split=split,selected=model==selected,**metrics(df.loc[idx,'y'].to_numpy(),preds[split])))
            idx=ev['test']; part=df.loc[idx,['date','station_id','city','y']].copy(); part['pred']=preds['test']; part['model']=model; part['mode']=mode
            part['selected']=model==selected; part['reference']=predictions['Persistence']['test']
            if model==selected:
                part['lower']=np.clip(part.pred-radius,0,500); part['upper']=np.clip(part.pred+radius,0,500)
                manifest['modes'][mode].update(interval_radius=radius,nominal_interval_coverage=.9,empirical_test_coverage=float(((part.y>=part.lower)&(part.y<=part.upper)).mean()),paired_rmse_gain=paired_week_bootstrap(part))
            all_predictions.append(part)
            for city,g in part.groupby('city'): all_group.append(dict(mode=mode,model=model,group_type='city',group=city,**metrics(g.y.to_numpy(),g.pred.to_numpy())))
            part['season']=pd.to_datetime(part.date).dt.month.map({12:'winter',1:'winter',2:'winter',3:'pre-monsoon',4:'pre-monsoon',5:'pre-monsoon',6:'monsoon',7:'monsoon',8:'monsoon',9:'monsoon',10:'post-monsoon',11:'post-monsoon'})
            for season,g in part.groupby('season'): all_group.append(dict(mode=mode,model=model,group_type='season',group=season,**metrics(g.y.to_numpy(),g.pred.to_numpy())))
        pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(all_group).to_csv(out/'group_metrics.csv',index=False)
        pd.concat(all_predictions).to_csv(out/'predictions.csv.gz',index=False)
        pd.DataFrame(seeds_metrics).to_csv(out/'seed_selection_metrics.csv',index=False)
        save(out,manifest)
        print(f'{mode}: selected {selected}; 2024 RMSE {metrics(df.loc[ev["test"],"y"].to_numpy(),predictions[selected]["test"])["RMSE"]:.2f}',flush=True)
    manifest.update(status='complete',ended_utc=datetime.now(timezone.utc).isoformat())
    save(out,manifest)
    print(pd.DataFrame(all_metrics).query("split=='test'").to_string(index=False),flush=True)

if __name__=='__main__': main()
