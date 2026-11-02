"""Evaluate a frozen 2023-2024 forecast once on quality-checked 2025 observations."""
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
from forecast_development import feature_table
from year_holdout import forecast_windows
from seq_models import make_net,_predict
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def scores(y,p):
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    return dict(n=len(y),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)))

def predict(artifact,development,aligned,rows,device='cpu'):
    if artifact['kind']=='persistence': return rows.aqi_l1.to_numpy()
    if artifact['kind']=='residual_tabular':
        return np.clip(rows.aqi_l1.to_numpy()+artifact['model'].predict(rows[artifact['features']]),0,500)
    if artifact['kind']!='gru_ensemble': raise ValueError('Unknown artifact kind')
    raw=forecast_windows(aligned,rows,artifact['window'])
    x=np.nan_to_num((raw-artifact['mu'])/artifact['sd']).astype(np.float32)
    predictions=[]
    for checkpoint in artifact['checkpoints']:
        net=make_net('GRU',artifact['n_in'])
        net.load_state_dict(torch.load(development/'models'/checkpoint['weight'],map_location='cpu',weights_only=True))
        net.to(device)
        predictions.append(np.clip(_predict(net,x)*checkpoint['target_std']+checkpoint['target_mean'],0,500))
    return np.mean(predictions,axis=0)

def paired_gain(rows):
    r=rows.copy(); r['model_error']=(r.y-r.pred)**2; r['base_error']=(r.y-r.reference)**2
    daily=r.groupby('date').agg(model=('model_error','sum'),base=('base_error','sum'),n=('y','size')).reindex(pd.date_range('2025-01-01','2025-12-31'),fill_value=0)
    blocks=daily.groupby(np.arange(len(daily))//7).sum().to_numpy(); blocks=blocks[blocks[:,2]>0]
    if not len(blocks): raise ValueError('Empty test')
    draws=blocks[np.random.default_rng(0).integers(0,len(blocks),(2000,len(blocks)))].sum(axis=1)
    gain=np.sqrt(draws[:,1]/draws[:,2])-np.sqrt(draws[:,0]/draws[:,2])
    return dict(gain=float(np.sqrt(daily.base.sum()/daily.n.sum())-np.sqrt(daily.model.sum()/daily.n.sum())),lower=float(np.quantile(gain,.025)),upper=float(np.quantile(gain,.975)),method='Paired seven-day blocks across all stations; longer temporal dependence may remain')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--development-run',default='2025-ready-v1'); ap.add_argument('--run-name',default='locked-2025-v1'); ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=ap.parse_args()
    for name in (a.development_run,a.run_name):
        if not name.replace('-','').replace('_','').isalnum(): ap.error('Invalid run name')
    development=ROOT/'results/forecast_development/runs'/a.development_run
    frozen=json.loads((development/'manifest.json').read_text())
    if frozen['status']!='frozen_awaiting_2025' or frozen['test_year_loaded']: raise ValueError('Development is not frozen')
    if sha256((ROOT/'src/models/forecast_development.py').read_bytes()).hexdigest()!=frozen['source_sha256']['forecast_development.py']: raise ValueError('Feature code changed since freezing')
    for file,expected in {**frozen['model_sha256'],'forecast_model.joblib':frozen['artifact_sha256']}.items():
        if sha256((development/file).read_bytes()).hexdigest()!=expected: raise ValueError(f'Frozen model changed: {file}')
    data=ROOT/'data/processed/multiyear_2023_2024_2025'; provenance=json.loads((data/'manifest.json').read_text()); path=data/'aligned_dataset.csv'
    if provenance['status']!='quality_checked' or provenance['protected_2023_2024_sha256']!=frozen['dataset_sha256'] or sha256(path.read_bytes()).hexdigest()!=provenance['dataset_sha256']: raise ValueError('Unchecked test data or altered development history')
    out=ROOT/'results/forecast_2025/runs'/a.run_name; out.mkdir(parents=True,exist_ok=False)
    # Reserving this directory prevents accidental overwrite/repeated evaluation.
    manifest=dict(status='evaluating',frozen_development_run=a.development_run,frozen_utc=frozen['frozen_utc'],started_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=provenance['dataset_sha256'],selected_model=frozen['selected_model'],limitations=frozen['limitations'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    aligned=pd.read_csv(path,parse_dates=['date']); df,_=feature_table(aligned)
    test=df[(df.date>='2025-01-01')&(df.date<'2026-01-01')].reset_index(drop=True)
    if len(test)<30: raise ValueError('Insufficient test rows')
    artifact=joblib.load(development/'forecast_model.joblib'); prediction=predict(artifact,development,aligned,test,a.device)
    r=test[['date','station_id','city','station_name','y']].copy(); r['pred']=prediction; r['reference']=test.aqi_l1
    radius=frozen['interval_radius']; r['lower']=np.clip(r.pred-radius,0,500); r['upper']=np.clip(r.pred+radius,0,500)
    metrics=pd.DataFrame([dict(model=frozen['selected_model'],**scores(r.y,r.pred)),dict(model='Persistence reference',**scores(r.y,r.reference))])
    groups=[]
    r['season']=r.date.dt.month.map({12:'winter',1:'winter',2:'winter',3:'pre-monsoon',4:'pre-monsoon',5:'pre-monsoon',6:'monsoon',7:'monsoon',8:'monsoon',9:'monsoon',10:'post-monsoon',11:'post-monsoon'})
    for group in ('city','season'):
        for name,g in r.groupby(group):
            for model,col in ((frozen['selected_model'],'pred'),('Persistence reference','reference')): groups.append(dict(group_type=group,group=name,model=model,**scores(g.y,g[col])))
    r.to_csv(out/'predictions.csv.gz',index=False); metrics.to_csv(out/'metrics.csv',index=False); pd.DataFrame(groups).to_csv(out/'group_metrics.csv',index=False)
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),rows=len(r),cities=int(r.city.nunique()),stations=int(r.station_id.nunique()),paired_rmse_gain=paired_gain(r),empirical_interval_coverage=float(((r.y>=r.lower)&(r.y<=r.upper)).mean()),nominal_interval_coverage=.9,evaluation_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(metrics.to_string(index=False),flush=True)

if __name__=='__main__': main()
