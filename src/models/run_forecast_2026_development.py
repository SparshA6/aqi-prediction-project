"""GPU weather/change candidates; freeze before any 2026 AQI labels are read."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from forecast_development import feature_table
from run_forecast_development import scores
from evaluate_forecast_2025 import predict

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/forecast_development/runs/2026-ready-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    data=ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv'
    weather=ROOT/'data/processed/forecast_weather/daily.csv'
    aligned=pd.read_csv(data,parse_dates=['date'])
    if set(aligned.date.dt.year)!={2023,2024,2025}: raise ValueError('2026 labels cannot enter development')
    w=pd.read_csv(weather,parse_dates=['date','forecast_available_before_utc'])
    w=w[w.date.dt.year<2026].copy()
    if (w.forecast_available_before_utc>w.date).any(): raise ValueError('Forecast issued after target starts')
    df,families=feature_table(aligned)
    df=df.merge(w,on=['city','date'],how='left',validate='many_to_one')
    df=df[df.date>='2024-01-01'].reset_index(drop=True)
    fc=[c for c in w if c.startswith('fc_')]
    idx={'train':np.flatnonzero(df.date<'2025-10-01'),
         'selection':np.flatnonzero((df.date>='2025-10-01')&(df.date<'2025-11-01')),
         'calibration':np.flatnonzero(df.date>='2025-11-01')}
    manifest=dict(status='training',protocol='Train Jan 2024–Sep 2025; select Oct 2025; calibrate Nov–Dec 2025; untouched 2026 Jan–Sep test',
        device='cuda',test_year_loaded=False,development_dataset_sha256=sha256(data.read_bytes()).hexdigest(),
        weather_dataset_sha256=sha256(weather.read_bytes()).hexdigest(),rows={k:len(i) for k,i in idx.items()},
        caveats=['2025 is now development data; the frozen 2025 benchmark is retained separately.',
                 'City-scale archived GFS forecasts use previous_day1 (24h lead); original ingestion latency is unverified.',
                 'Past station AQI is assumed received by issue time. Missing forecasts remain missing.',
                 '2026 scores must not be used to tune or select candidates.'])
    def save(): (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save(); candidates={'Persistence':{k:df.loc[i,'aqi_l1'].to_numpy() for k,i in idx.items() if k!='train'}}
    artifacts={'Persistence':dict(kind='persistence',features=['aqi_l1'])}
    frozen=ROOT/'results/forecast_development/runs/2025-ready-v1'
    old=joblib.load(frozen/'forecast_model.joblib')
    candidates['Frozen GRU-28days']={k:predict(old,frozen,aligned,df.loc[i],'cuda') for k,i in idx.items() if k!='train'}
    artifacts['Frozen GRU-28days']=dict(kind='frozen_gru',development_run='2025-ready-v1')
    tr=idx['train']; target=(df.loc[tr,'y']-df.loc[tr,'aqi_l1']).to_numpy()
    for family,features in [('history',families['long']),('forecast-weather',families['long']+fc)]:
        for depth in (3,5):
            name=f'XGBoost-change-{family}-d{depth}'; print('GPU training '+name,flush=True)
            estimator=XGBRegressor(n_estimators=600,max_depth=depth,learning_rate=.035,subsample=.85,colsample_bytree=.9,
                reg_lambda=5,min_child_weight=15,tree_method='hist',device='cuda',random_state=2,n_jobs=2)
            estimator.fit(df.loc[tr,features],target)
            candidates[name]={k:np.clip(df.loc[i,'aqi_l1'].to_numpy()+estimator.predict(df.loc[i,features]),0,500) for k,i in idx.items() if k!='train'}
            artifacts[name]=dict(kind='residual_tabular',model=estimator,features=features)
    metrics=[]; pred=[]
    for name,values in candidates.items():
        for split,p in values.items():
            i=idx[split]; metrics.append(dict(model=name,split=split,**scores(df.loc[i,'y'],p)))
            r=df.loc[i,['date','station_id','city','y','aqi_l1']].copy();r['pred']=p;r['model']=name;r['split']=split;pred.append(r)
    m=pd.DataFrame(metrics); chosen=m[m.split=='selection'].sort_values(['RMSE','model']).iloc[0].model
    errors=np.abs(df.loc[idx['calibration'],'y'].to_numpy()-candidates[chosen]['calibration'])
    radius=float(np.quantile(errors,min(1,np.ceil((len(errors)+1)*.9)/len(errors)),method='higher'))
    artifact=artifacts[chosen]; artifact.update(selected=chosen,interval_radius=radius)
    joblib.dump(artifact,OUT/'forecast_model.joblib')
    m.to_csv(OUT/'development_metrics.csv',index=False);pd.concat(pred).to_csv(OUT/'development_predictions.csv.gz',index=False)
    manifest.update(status='frozen_awaiting_2026',selected_model=chosen,interval_radius=radius,
        artifact_sha256=sha256((OUT/'forecast_model.joblib').read_bytes()).hexdigest(),frozen_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    save();print(m.to_string(index=False),flush=True);print('Frozen '+chosen+'; 2026 AQI has not been loaded',flush=True)

if __name__=='__main__':main()
