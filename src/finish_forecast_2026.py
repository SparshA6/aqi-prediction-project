"""Wait for complete Jan–Sep inputs, quality-check them, score a frozen model once."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
import time
import sys
import argparse
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib
from paths import ROOT,STATIONS_CSV
from combine_with_aqi import FILENAME_RE,MONTH_NAMES,daily_aggregate,parse_cpcb_month_file
sys.path.insert(0,str(ROOT/'src/models'))
from forecast_development import feature_table,WEATHER_RAW
from run_forecast_development import scores
from evaluate_forecast_2025 import predict
from forecast_diagnostics import summarize

DEV=ROOT/'results/forecast_development/runs/2026-ready-v1'
OUT=ROOT/'results/forecast_2026/runs/locked-2026-v1'
STAGE=ROOT/'data/processed/forecast_2026'

def files_for(row):
    relative=Path(row.data_dir).relative_to(Path('data/raw/cpcb'))
    result=[]
    for p in (ROOT/'data/raw/cpcb_2026'/relative).glob('*.xlsx'):
        m=FILENAME_RE.search(p.name)
        if m and (m.group('year_first') or m.group('year_second'))=='2026':
            month=MONTH_NAMES.index((m.group('month_a') or m.group('month_b')).title())+1
            if month<=9: result.append((month,p))
    return result

def readiness():
    missing=[]
    for row in pd.read_csv(STATIONS_CSV).itertuples():
        files=files_for(row)
        if {m for m,p in files}!=set(range(1,10)) or len(files)!=9:missing.append(row.station_id)
    return missing

def completed_retry_exclusions(log):
    """Use download availability only; do not inspect target scores to choose sites."""
    text=Path(log).read_text(encoding='utf-8',errors='replace')
    if '                    COMPLETE' not in text: return None
    exclusions=[]
    for row in pd.read_csv(STATIONS_CSV).itertuples():
        files=files_for(row)
        if {m for m,p in files}==set(range(1,10)) and len(files)==9:continue
        if files:raise ValueError('Partial or duplicate exports require repair: '+row.station_id)
        signature=f'[{row.city}] {row.station_name}: 0/9 months'
        if signature not in text:raise ValueError('Missing station was not retried: '+row.station_id)
        exclusions.append(dict(station_id=row.station_id,station_name=row.station_name,city=row.city,
            reason='No monthly exports after completed retry; portal availability or request failure unresolved',
            retry_log_sha256=sha256(Path(log).read_bytes()).hexdigest()))
    return exclusions

def stage(exclusions=None):
    STAGE.mkdir(parents=True,exist_ok=True);(STAGE/'cache').mkdir(exist_ok=True)
    frames=[];hashes={};audit=[]
    for row in pd.read_csv(STATIONS_CSV).itertuples():
        if row.station_id in {r['station_id'] for r in exclusions or []}:continue
        files=[p for m,p in files_for(row)]
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
        fingerprint=sha256(json.dumps([sources,sha256(Path(__file__).read_bytes()).hexdigest(),row._asdict()],sort_keys=True).encode()).hexdigest()
        cp=STAGE/'cache'/f'{row.station_id}.csv';mp=cp.with_suffix('.json')
        saved=json.loads(mp.read_text()) if mp.exists() else {}
        if cp.exists() and saved.get('fingerprint')==fingerprint and saved.get('sha256')==sha256(cp.read_bytes()).hexdigest():
            daily=pd.read_csv(cp,parse_dates=['date'])
        else:
            hourly=pd.concat([parse_cpcb_month_file(p) for p in files],ignore_index=True)
            hourly.aqi=pd.to_numeric(hourly.aqi,errors='raise')
            if ((hourly.aqi.dropna()<0)|(hourly.aqi.dropna()>500)).any():raise ValueError('AQI outside range: '+row.station_id)
            dup=hourly.duplicated(['date','hour_ist'],keep=False)
            if hourly[dup].groupby(['date','hour_ist']).aqi.nunique().gt(1).any():raise ValueError('Conflicting readings: '+row.station_id)
            daily=daily_aggregate(hourly.drop_duplicates(['date','hour_ist']))
            daily=daily[(daily.date>='2026-01-01')&(daily.date<'2026-10-01')].copy()
            daily['station_id']=row.station_id;daily['city']=row.city;daily['station_name']=row.station_name
            daily['station_lat']=row.lat;daily['station_lon']=row.lon
            daily['aqi_flat_day']=(daily.AQI_min==daily.AQI_max)&(daily.AQI_max<500)
            daily.to_csv(cp,index=False)
            mp.write_text(json.dumps(dict(fingerprint=fingerprint,sha256=sha256(cp.read_bytes()).hexdigest())))
        frames.append(daily);hashes.update(sources)
        audit.append(dict(station_id=row.station_id,city=row.city,rows=len(daily),usable=int((daily.AQI_eod.notna()&~daily.aqi_flat_day).sum())))
        print('Checked '+row.station_id,flush=True)
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(['station_id','date']).any():raise ValueError('Duplicate station dates')
    frame.to_csv(STAGE/'aligned_dataset.csv',index=False);pd.DataFrame(audit).to_csv(STAGE/'coverage.csv',index=False)
    pd.DataFrame(exclusions or [],columns=['station_id','station_name','city','reason','retry_log_sha256']).to_csv(STAGE/'excluded_stations.csv',index=False)
    (STAGE/'manifest.json').write_text(json.dumps(dict(status='quality_checked',stations=int(frame.station_id.nunique()),expected_stations=len(pd.read_csv(STATIONS_CSV)),excluded_stations=exclusions or [],dataset_sha256=sha256((STAGE/'aligned_dataset.csv').read_bytes()).hexdigest(),source_sha256=hashes,
        caveats=['>=12 hourly readings required; flat days excluded.','September 30 EOD requires October input, not collected.','No observed weather or satellite values are fabricated.']),indent=2))
    return frame

def evaluate(fresh):
    frozen=json.loads((DEV/'manifest.json').read_text())
    if json.loads((DEV/'verification.json').read_text()).get('status')!='passed':raise ValueError('Development verification missing')
    artifact_path=DEV/'forecast_model.joblib'
    if frozen['status']!='frozen_awaiting_2026' or sha256(artifact_path.read_bytes()).hexdigest()!=frozen['artifact_sha256']:raise ValueError('Model freeze failed')
    metrics=pd.read_csv(DEV/'development_metrics.csv')
    selected=metrics[metrics.split=='selection'].sort_values(['RMSE','model']).iloc[0].model
    if selected!=frozen['selected_model']:raise ValueError('Selection disagrees with frozen manifest')
    historical_path=ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv'
    if sha256(historical_path.read_bytes()).hexdigest()!=frozen['development_dataset_sha256']:raise ValueError('Historical inputs changed')
    aligned=pd.concat([pd.read_csv(historical_path,parse_dates=['date']),fresh],ignore_index=True)
    for col in WEATHER_RAW:
        if col not in aligned:aligned[col]=np.nan
    df,f=feature_table(aligned);df=df[(df.date>='2026-01-01')&(df.date<'2026-10-01')].reset_index(drop=True)
    weather_path=ROOT/'data/processed/forecast_weather/daily.csv'
    if sha256(weather_path.read_bytes()).hexdigest()!=frozen['weather_dataset_sha256']:raise ValueError('Forecast archive changed')
    weather=pd.read_csv(weather_path,parse_dates=['date','forecast_available_before_utc'])
    if (weather.forecast_available_before_utc>weather.date).any():raise ValueError('Late forecast input')
    df=df.merge(weather,on=['city','date'],how='left',validate='many_to_one')
    artifact=joblib.load(artifact_path)
    def predictions(rows):
        if artifact['kind']=='persistence':return rows.aqi_l1.to_numpy()
        if artifact['kind']=='frozen_gru':
            dev=DEV.parent/artifact['development_run'];old=joblib.load(dev/'forecast_model.joblib')
            return predict(old,dev,aligned,rows,'cpu')
        return np.clip(rows.aqi_l1.to_numpy()+artifact['model'].predict(rows[artifact['features']]),0,500)
    p=predictions(df);base=df.aqi_l1.to_numpy();y=df.y.to_numpy()
    old_dev=ROOT/'results/forecast_development/runs/2025-ready-v1'
    old_manifest=json.loads((old_dev/'manifest.json').read_text())
    if sha256((old_dev/'forecast_model.joblib').read_bytes()).hexdigest()!=old_manifest['artifact_sha256']:raise ValueError('GRU artifact changed')
    old=joblib.load(old_dev/'forecast_model.joblib')
    for checkpoint in old['checkpoints']:
        key=str(Path('models')/checkpoint['weight'])
        if sha256((old_dev/key).read_bytes()).hexdigest()!=old_manifest['model_sha256'][key]:raise ValueError('GRU checkpoint changed')
    gru=predict(old,old_dev,aligned,df,'cuda')
    result=df[['date','station_id','station_name','city','y']].copy();result['pred']=p;result['reference']=base;result['gru_pred']=gru
    result['lower']=np.maximum(0,p-frozen['interval_radius']);result['upper']=np.minimum(500,p+frozen['interval_radius'])
    OUT.mkdir(parents=True,exist_ok=False);result.to_csv(OUT/'predictions.csv.gz',index=False)
    overall=pd.DataFrame([dict(model=selected,**scores(y,p)),dict(model='Persistence reference',**scores(y,base)),dict(model='Frozen GRU-28days',**scores(y,gru))]);overall.to_csv(OUT/'metrics.csv',index=False)
    group=[]
    for city,g in result.groupby('city'):
        for name,key in [(selected,'pred'),('Persistence reference','reference'),('Frozen GRU-28days','gru_pred')]:group.append(dict(city=city,model=name,**scores(g.y,g[key])))
    pd.DataFrame(group).to_csv(OUT/'city_metrics.csv',index=False)
    loaded=pd.read_csv(OUT/'predictions.csv.gz');recomputed=scores(loaded.y,loaded.pred)
    if not np.isclose(recomputed['RMSE'],overall.iloc[0].RMSE,atol=1e-9):raise ValueError('Independent score mismatch')
    sample=np.linspace(0,len(df)-1,min(80,len(df)),dtype=int)
    if not np.allclose(predictions(df.iloc[sample]),p[sample],atol=1e-3):raise ValueError('Reload mismatch')
    if not np.allclose(predict(old,old_dev,aligned,df.iloc[sample],'cpu'),gru[sample],atol=1e-3):raise ValueError('CPU GRU reload mismatch')
    weekly=pd.DataFrame(dict(week=((pd.to_datetime(result.date)-pd.Timestamp('2026-01-01')).dt.days//7),
        model=(p-y)**2,reference=(base-y)**2,n=1)).groupby('week').sum()
    rng=np.random.default_rng(42);gains=[]
    for _ in range(2000):
        draw=weekly.iloc[rng.integers(0,len(weekly),len(weekly))].sum()
        gains.append(float(np.sqrt(draw.reference/draw.n)-np.sqrt(draw.model/draw.n)))
    gain=dict(gain=float(overall.iloc[1].RMSE-overall.iloc[0].RMSE),
        lower=float(np.quantile(gains,.025)),upper=float(np.quantile(gains,.975)),method='paired calendar-week block bootstrap, 2000 draws')
    weekly['reference']=pd.Series((gru-y)**2).groupby(((pd.to_datetime(result.date)-pd.Timestamp('2026-01-01')).dt.days//7)).sum()
    gains=[];rng=np.random.default_rng(42)
    for _ in range(2000):
        draw=weekly.iloc[rng.integers(0,len(weekly),len(weekly))].sum()
        gains.append(float(np.sqrt(draw.reference/draw.n)-np.sqrt(draw.model/draw.n)))
    gru_gain=dict(gain=float(overall.iloc[2].RMSE-overall.iloc[0].RMSE),lower=float(np.quantile(gains,.025)),upper=float(np.quantile(gains,.975)))
    diagnostics=[]
    for city,g in [('all',result),*list(result.groupby('city'))]:
        for name,key in [(selected,'pred'),('Frozen GRU-28days','gru_pred')]:
            r=g[['y','reference',key]].rename(columns={key:'pred'})
            diagnostics.append(dict(city=city,model=name,**summarize(r)))
    spikes=pd.DataFrame(diagnostics);spikes.to_csv(OUT/'spike_metrics.csv',index=False)
    new_spike=spikes[(spikes.city=='all')&(spikes.model==selected)].iloc[0]
    old_spike=spikes[(spikes.city=='all')&(spikes.model=='Frozen GRU-28days')].iloc[0]
    # Predeclared promotion: improvement over both comparators, with no loss of
    # sharp-change recall or precision. Per-city regressions remain visible.
    approved=bool(gain['lower']>0 and gru_gain['lower']>0 and new_spike.recall>=old_spike.recall and new_spike.precision>=old_spike.precision)
    staged=json.loads((STAGE/'manifest.json').read_text())
    manifest=dict(status='complete',selected_model=selected,development_run=DEV.name,test_year=2026,period='Jan–Sep 2026',rows=len(result),
        interval_coverage=float(((y>=result.lower)&(y<=result.upper)).mean()),development_artifact_sha256=frozen['artifact_sha256'],
        paired_rmse_gain=gain,
        paired_gru_rmse_gain=gru_gain,recommended=approved,
        promotion_rule='Positive lower 95% paired weekly RMSE gain vs persistence and GRU; sharp-change recall and precision >= GRU at threshold 50',
        stations=int(result.station_id.nunique()),expected_stations=staged['expected_stations'],excluded_stations=staged['excluded_stations'],
        frozen_utc=frozen['frozen_utc'],evaluated_utc=datetime.now(timezone.utc).isoformat(),limitations=frozen['caveats'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(status='awaiting_independent_check',metrics_recomputed=True,reload_checked=True,test_excluded_from_fit=True),indent=2))
    print(overall.to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',type=Path,required=True);ap.add_argument('--retry-log',type=Path);args=ap.parse_args();args.outputs.mkdir(exist_ok=True,parents=True)
    status_path=args.outputs/'Forecast_2026_Status.json'
    def status(value,missing=0,error=None):
        payload=dict(status=value,incomplete_stations=missing,error=error,updated_utc=datetime.now(timezone.utc).isoformat())
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(status_path)
    start=time.monotonic()
    try:
        exclusions=[]
        while True:
            missing=readiness();status('waiting_for_download',len(missing))
            if not missing:break
            if args.retry_log and args.retry_log.exists():
                decision=completed_retry_exclusions(args.retry_log)
                if decision is not None:
                    exclusions=decision;status('retry_complete_with_coverage_gaps',len(exclusions));break
            if time.monotonic()-start>24*3600:status('needs_resume',len(missing));return
            print(f'Waiting: {len(missing)} stations incomplete',flush=True);time.sleep(60)
        if OUT.exists() and (OUT/'verification.json').exists():status('complete');return
        status('checking_data');fresh=stage(exclusions);status('evaluating_frozen_model');evaluate(fresh)
        subprocess.run([sys.executable,'-B','src/models/verify_forecast_2026.py'],cwd=ROOT,check=True)
        for name in ['metrics.csv','city_metrics.csv','spike_metrics.csv','manifest.json','verification.json','predictions.csv.gz']:shutil.copy2(OUT/name,args.outputs/('Forecast_2026_'+name))
        shutil.copy2(STAGE/'excluded_stations.csv',args.outputs/'Forecast_2026_Excluded_Stations.csv')
        status('complete')
    except Exception as exc:status('failed',error=str(exc));raise

if __name__=='__main__':main()
