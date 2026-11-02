"""Collect input snapshots and record immutable forecasts before target UTC midnight.

This is a prospective recorder, separate from frozen historical evaluation.
No target-day labels are loaded by record().
"""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import argparse
import json
import sys
import numpy as np
import pandas as pd
import requests
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/models'))
from ensemble_followthrough import load_frozen,GRU,WEATHER,blend
from evaluate_forecast_2025 import predict
from forecast_development import feature_table,WEATHER_RAW
from prepare_prospective_inputs import VARIABLES
from combine_with_aqi import parse_cpcb_month_file,daily_aggregate

OUT=ROOT/'results/prospective'
RAW=ROOT/'data/raw/cpcb_prospective'

def now_utc():return pd.Timestamp(datetime.now(timezone.utc))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def stamp():return now_utc().strftime('%Y%m%dT%H%M%S%fZ')
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def target_day(value):
    day=pd.Timestamp(value)
    if day.tzinfo is not None or day!=day.normalize():raise ValueError('Use an unzoned YYYY-MM-DD UTC target date')
    if not pd.Timestamp('2026-10-13')<=day<=pd.Timestamp('2026-12-31'):raise ValueError('Target outside reserved period')
    return day

def collect_weather(target):
    """Save raw response snapshots even if not all earlier runs are available."""
    target=target_day(target);retrieved=now_utc();directory=OUT/'weather'/stamp();directory.mkdir(parents=True)
    cities=pd.read_csv(ROOT/'config/stations.csv').groupby('city')[['lat','lon']].mean().reset_index()
    records=[];sources={};retrieval_audit=[];single_run_audit=[]
    run_time=(target-pd.Timedelta(days=1)+pd.Timedelta(hours=12)).tz_localize('UTC')
    run_name=run_time.strftime('%Y-%m-%dT%H:%M')
    for city in cities.itertuples():
        row=dict(date=str(target.date()),city=city.city,status='unavailable',available_hours=0)
        audit=dict(city=city.city,request_started_utc=now_utc().isoformat(),
                   response_received_utc=None,provider_http_date=None,
                   provider_run_id=None,raw_sha256=None)
        run_audit=dict(city=city.city,run_initialized_utc=run_time.isoformat(),
                       status='not_attempted',response_received_utc=None,
                       provider_http_date=None,raw_sha256=None)
        try:
            response=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params=dict(latitude=city.lat,longitude=city.lon,
                start_date=str(target.date()),end_date=str(target.date()),timezone='UTC',models='gfs_seamless',
                temperature_unit='celsius',wind_speed_unit='kmh',precipitation_unit='mm',
                hourly=','.join(v+'_previous_day1' for v in VARIABLES)),timeout=40)
            audit['response_received_utc']=now_utc().isoformat()
            audit['provider_http_date']=getattr(response,'headers',{}).get('Date')
            audit['request_url']=getattr(response,'url',None)
            response.raise_for_status();data=response.json();raw_path=directory/(city.city+'.json')
            raw_path.write_text(json.dumps(data),encoding='utf-8');sources[raw_path.name]=digest(raw_path)
            audit['raw_sha256']=sources[raw_path.name]
            audit['provider_run_id']=data.get('run') or data.get('model_run')
            h=pd.DataFrame(data['hourly']);hours=pd.to_datetime(h.time,utc=True)
            expected=pd.date_range(target,periods=24,freq='h',tz='UTC')
            if hours.duplicated().any() or set(hours)!=set(expected):raise ValueError('Incomplete or duplicate hourly timestamps')
            issue=hours-pd.Timedelta(hours=24)
            known=issue<=retrieved
            matrix=np.column_stack([pd.to_numeric(h[v+'_previous_day1'],errors='coerce') for v in VARIABLES])
            finite=np.isfinite(matrix).all(axis=1)&known.to_numpy()
            row['available_hours']=int(finite.sum());row['forecast_available_before_utc']=issue.max().isoformat()
            if finite.all() and issue.max()<target.tz_localize('UTC'):
                row['status']='ready'
                for j,v in enumerate(VARIABLES):row['fc_'+v]=float(matrix[:,j].sum() if v=='precipitation' else matrix[:,j].mean())
            else:row['status']='waiting_for_complete_earlier_runs'
            # Independent exact-run archive for audit only. Its values are not model inputs.
            if row['status']=='ready' and now_utc()>=run_time+pd.Timedelta(hours=6):
                run_audit['status']='unavailable'
                run_audit['request_started_utc']=now_utc().isoformat()
                try:
                    run_response=requests.get('https://single-runs-api.open-meteo.com/v1/forecast',params=dict(
                        latitude=city.lat,longitude=city.lon,start_date=str(target.date()),end_date=str(target.date()),
                        timezone='UTC',models='gfs_seamless',run=run_name,temperature_unit='celsius',
                        wind_speed_unit='kmh',precipitation_unit='mm',hourly=','.join(VARIABLES)),timeout=20)
                    run_audit['response_received_utc']=now_utc().isoformat()
                    run_audit['provider_http_date']=getattr(run_response,'headers',{}).get('Date')
                    run_audit['request_url']=getattr(run_response,'url',None)
                    run_response.raise_for_status();run_data=run_response.json()
                    run_path=directory/(city.city+'.single_run.json')
                    run_path.write_text(json.dumps(run_data),encoding='utf-8')
                    run_audit['raw_sha256']=digest(run_path)
                    rh=pd.DataFrame(run_data['hourly']);rt=pd.to_datetime(rh.time,utc=True)
                    complete=not rt.duplicated().any() and set(rt)==set(expected)
                    complete=complete and all(v in rh and pd.to_numeric(rh[v],errors='coerce').notna().all() for v in VARIABLES)
                    run_audit['status']='complete_exact_run' if complete else 'incomplete_exact_run'
                except (requests.RequestException,ValueError,KeyError) as e:
                    run_audit['error']=str(e)[:250]
            elif row['status']=='ready':
                run_audit['status']='run_not_yet_due'
        except (requests.RequestException,ValueError,KeyError) as e:row['error']=str(e)[:250]
        audit['status']=row['status'];retrieval_audit.append(audit)
        single_run_audit.append(run_audit)
        records.append(row);print(city.city+': '+row['status'],flush=True)
    pd.DataFrame(records).to_csv(directory/'daily.csv',index=False)
    write_json(directory/'retrieval_audit.json',retrieval_audit)
    write_json(directory/'single_run_audit.json',single_run_audit)
    write_json(directory/'manifest.json',dict(status='snapshot_collected',target_date=str(target.date()),retrieved_utc=retrieved.isoformat(),
        lead_hours=24,model='gfs_seamless',dataset_sha256=digest(directory/'daily.csv'),source_sha256=sources,
        retrieval_audit_sha256=digest(directory/'retrieval_audit.json'),
        exact_run_audit_sha256=digest(directory/'single_run_audit.json'),exact_run_initialized_utc=run_time.isoformat(),
        run_provenance='provider_run_id_when_returned; otherwise unavailable from previous-runs response',
        complete_cities=sum(r['status']=='ready' for r in records),expected_cities=len(cities),
        note='Each raw response has request/response timestamps and an HTTP Date when supplied. The named single run is an independent audit only; it is not claimed to be the source of every previous_day1 value. Run initialization is not publication time.'))
    return directory

def collect_ground():
    stations=pd.read_csv(ROOT/'config/stations.csv');audit=[];frames=[];hashes={}
    for station in stations.itertuples():
        relative=Path(station.data_dir).relative_to(Path('data/raw/cpcb'))
        files=sorted((RAW/relative).glob('*.xlsx'))
        row=dict(station_id=station.station_id,city=station.city,files=len(files),usable_days=0,latest_observed=None)
        if files:
            try:
                h=pd.concat([parse_cpcb_month_file(p) for p in files],ignore_index=True)
                h.aqi=pd.to_numeric(h.aqi,errors='raise')
                if ((h.aqi.dropna()<0)|(h.aqi.dropna()>500)).any():raise ValueError('AQI outside 0-500')
                if h.groupby(['date','hour_ist']).aqi.nunique().gt(1).any():raise ValueError('Conflicting duplicates')
                d=daily_aggregate(h.drop_duplicates(['date','hour_ist']))
                d=d[d.date>='2026-10-01'].copy()
                d['station_id']=station.station_id;d['station_name']=station.station_name;d['city']=station.city
                d['station_lat']=station.lat;d['station_lon']=station.lon
                d['aqi_flat_day']=(d.AQI_min==d.AQI_max)&(d.AQI_max<500)
                for col in WEATHER_RAW:d[col]=np.nan
                valid=d.AQI_eod.notna()&~d.aqi_flat_day
                row['usable_days']=int(valid.sum())
                if valid.any():row['latest_observed']=str(d.loc[valid,'date'].max().date())
                frames.append(d);hashes.update({str(p):digest(p) for p in files})
            except (ValueError,KeyError,OSError) as e:row['error']=str(e)[:250]
        audit.append(row)
    directory=OUT/'ground'/stamp();directory.mkdir(parents=True)
    pd.DataFrame(audit).to_csv(directory/'coverage.csv',index=False)
    # Ground snapshots may contain observed labels; recorder truncates them before feature construction.
    pd.concat(frames,ignore_index=True).to_csv(directory/'aligned_dataset.csv',index=False) if frames else None
    write_json(directory/'manifest.json',dict(status='snapshot_checked',captured_utc=now_utc().isoformat(),source_sha256=hashes,
        dataset_sha256=digest(directory/'aligned_dataset.csv') if frames else None,stations_with_exports=sum(r['files']>0 for r in audit),
        expected_stations=len(stations),usable_station_days=sum(r['usable_days'] for r in audit),failed_stations=[r for r in audit if r.get('error')]))
    return directory

def latest(kind):
    paths=sorted((OUT/kind).glob('*/manifest.json'))
    return paths[-1].parent if paths else None

def feature_rows(history,stations,target):
    """Target scaffolding provides metadata only, never a target observation."""
    history=history[pd.to_datetime(history.date)<target].copy()
    d=stations.rename(columns={'lat':'station_lat','lon':'station_lon'}).copy();d['date']=target
    d['AQI_eod']=0.;d['aqi_flat_day']=False  # discarded y; feature_table requires an eligible row to construct lags
    for col in WEATHER_RAW:d[col]=np.nan
    all_rows=pd.concat([history,d],ignore_index=True,sort=False)
    rows,_=feature_table(all_rows)
    return history,rows[rows.date==target].drop(columns='y').copy()

def validate_capture(target,now,weather,manifest):
    cutoff=target.tz_localize('UTC')
    if now>=cutoff:raise ValueError('Deadline passed: cannot create a prospective record for this target')
    if pd.Timestamp(manifest['retrieved_utc'])>=cutoff:raise ValueError('Weather captured after target began')
    if manifest.get('target_date')!=str(target.date()):raise ValueError('Weather snapshot is for another day')
    issue=pd.to_datetime(weather.forecast_available_before_utc,utc=True)
    if (issue>=cutoff).any() or (issue>now).any():raise ValueError('Weather issue time is not available before target')

def record(target):
    target=target_day(target);started=now_utc();frozen=load_frozen()
    if started>=target.tz_localize('UTC'):raise ValueError('Deadline passed: prospective logging refuses backdating')
    destination=OUT/'ledger'/str(target.date())
    if destination.exists():raise ValueError('A record already exists; recorded forecasts cannot be overwritten')
    wd=latest('weather');gd=latest('ground')
    if not wd or not gd:raise ValueError('Collect weather and October ground snapshots first')
    wm=json.loads((wd/'manifest.json').read_text());gm=json.loads((gd/'manifest.json').read_text())
    if digest(wd/'daily.csv')!=wm['dataset_sha256']:raise ValueError('Weather hash changed')
    if wm.get('retrieval_audit_sha256') and digest(wd/'retrieval_audit.json')!=wm['retrieval_audit_sha256']:
        raise ValueError('Weather retrieval audit hash changed')
    if wm.get('exact_run_audit_sha256') and digest(wd/'single_run_audit.json')!=wm['exact_run_audit_sha256']:
        raise ValueError('Exact-run audit hash changed')
    weather=pd.read_csv(wd/'daily.csv');weather=weather[weather.status=='ready'].copy()
    if weather.empty:raise ValueError('No city has all 24 eligible earlier-issued hourly forecasts yet')
    validate_capture(target,started,weather,wm)
    if not gm['dataset_sha256'] or digest(gd/'aligned_dataset.csv')!=gm['dataset_sha256']:raise ValueError('No valid ground snapshot')
    if pd.Timestamp(gm['captured_utc'])>=target.tz_localize('UTC'):raise ValueError('Ground captured after forecast deadline')
    stations=pd.read_csv(ROOT/'config/stations.csv')
    paths=[ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv',ROOT/'data/processed/forecast_2026/aligned_dataset.csv',gd/'aligned_dataset.csv']
    history=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in paths],ignore_index=True,sort=False)
    history=history[history.date<target]
    if history.duplicated(['station_id','date']).any():raise ValueError('Duplicate history dates')
    history,rows=feature_rows(history,stations,target)
    rows=rows.merge(weather[['city']+[c for c in weather if c.startswith('fc_')]],on='city',validate='many_to_one')
    artifact=joblib.load(WEATHER/'forecast_model.joblib')
    fc=[c for c in artifact['features'] if c.startswith('fc_')];rows=rows.dropna(subset=fc)
    if rows.empty:raise ValueError('No station has eligible prior-day AQI and complete forecast weather')
    artifact['model'].set_params(device='cpu')
    g=predict(joblib.load(GRU/'forecast_model.joblib'),GRU,history,rows,'cpu')
    w=predict(artifact,WEATHER,history,rows,'cpu')
    r=rows[['date','station_id','station_name','city','aqi_l1','aqi_count28']].copy()
    r['gru_pred']=g;r['weather_pred']=w;r['pred']=blend(g,w,frozen['weather_weight'])
    r['lower']=np.maximum(0,r.pred-frozen['interval_radius']);r['upper']=np.minimum(500,r.pred+frozen['interval_radius'])
    finished=now_utc()
    if finished>=target.tz_localize('UTC'):raise ValueError('Inference finished after deadline; no prospective record created')
    destination.mkdir(parents=True,exist_ok=False)
    r.to_csv(destination/'predictions.csv',index=False)
    write_json(destination/'manifest.json',dict(status='recorded_before_target',target_date=str(target.date()),recorded_utc=finished.isoformat(),
        rows=len(r),stations=len(r),predictions_sha256=digest(destination/'predictions.csv'),
        frozen_protocol_sha256=digest(ROOT/'results/ensemble/runs/prospective-oct13-v1/manifest.json'),
        weather_manifest=str(wd/'manifest.json'),weather_manifest_sha256=digest(wd/'manifest.json'),
        ground_manifest=str(gd/'manifest.json'),ground_manifest_sha256=digest(gd/'manifest.json'),
        earlier_history_sha256={str(p):digest(p) for p in paths},target_labels_used=False,
        note='Local timestamps and hashes document this capture; they are not third-party timestamp certification.'))
    print('Recorded',len(r),'station forecasts for',target.date())
    return destination

def refresh_status():
    frozen=load_frozen();stations=pd.read_csv(ROOT/'config/stations.csv');gd=latest('ground');wd=latest('weather')
    gm=json.loads((gd/'manifest.json').read_text()) if gd else {}
    wm=json.loads((wd/'manifest.json').read_text()) if wd else {}
    entries=[]
    for path in sorted((OUT/'ledger').glob('*/manifest.json')):
        m=json.loads(path.read_text());m['integrity_valid']=digest(path.parent/'predictions.csv')==m['predictions_sha256']
        entries.append({k:m[k] for k in ['target_date','recorded_utc','rows','status','integrity_valid']})
    city=[]
    coverage=pd.read_csv(gd/'coverage.csv') if gd else pd.DataFrame()
    for name,s in stations.groupby('city'):
        group=coverage[coverage.city==name] if len(coverage) else pd.DataFrame()
        city.append(dict(city=name,expected_stations=len(s),stations_with_exports=int((group.files>0).sum()) if len(group) else 0,
            usable_station_days=int(group.usable_days.sum()) if len(group) else 0,
            latest_observed=max(group.latest_observed.dropna(),default=None) if len(group) else None))
    data=dict(status='tracking',updated_utc=now_utc().isoformat(),period=frozen['prospective_period'],
        stations_with_exports=gm.get('stations_with_exports',0),expected_stations=len(stations),
        usable_station_days=gm.get('usable_station_days',0),ground_captured_utc=gm.get('captured_utc'),
        weather_target=wm.get('target_date'),weather_complete_cities=wm.get('complete_cities',0),expected_cities=len(city),
        weather_captured_utc=wm.get('retrieved_utc'),recorded_days=len(entries),records=entries,cities=city,
        downloader_waiting_captcha=False,final_evaluation_date='2027-01-01',
        note='October counts include pre-test history. Only immutable before-target records count as prospective forecasts. No final test score yet.')
    log=ROOT/'logs/prospective_oct13.log'
    if log.exists():
        text=log.read_text(encoding='utf-8',errors='replace')
        data['downloader_waiting_captcha']='Auto mode: solve' in text and 'CAPTCHA solved.' not in text and 'No CAPTCHA appeared' not in text
        data['download_completed']='                    COMPLETE' in text
    write_json(OUT/'status.json',data);return data

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['weather','ground','record','status'])
    ap.add_argument('--target',default=str((now_utc().tz_localize(None).normalize()+pd.Timedelta(days=1)).date()));a=ap.parse_args()
    try:
        if a.action=='weather':collect_weather(a.target)
        elif a.action=='ground':collect_ground()
        elif a.action=='record':record(a.target)
        print(json.dumps(refresh_status(),indent=2))
    except (ValueError,FileNotFoundError) as e:
        refresh_status();print(str(e),file=sys.stderr);sys.exit(2)

if __name__=='__main__':main()
