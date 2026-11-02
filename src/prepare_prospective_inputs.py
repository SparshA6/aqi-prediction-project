"""Prepare new inputs without overwriting historical benchmarks or forecast archives."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import argparse
import json
import sys
import numpy as np
import pandas as pd
import requests
from paths import ROOT,STATIONS_CSV
from combine_with_aqi import parse_cpcb_month_file,daily_aggregate
sys.path.insert(0,str(ROOT/'src/models'))
from forecast_development import WEATHER_RAW

VARIABLES=['temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation','surface_pressure']

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()

def aggregate_archive(data,city):
    h=pd.DataFrame(data['hourly']);h['valid_hour']=pd.to_datetime(h.time,utc=True)
    if h.valid_hour.duplicated().any():raise ValueError('Duplicate archive hours')
    h['date']=h.valid_hour.dt.tz_localize(None).dt.floor('D')
    rows=[]
    for day,g in h.groupby('date'):
        expected=pd.date_range(day,periods=24,freq='h',tz='UTC')
        complete=set(g.valid_hour)==set(expected)
        issue=(g.valid_hour-pd.Timedelta(hours=24)).max()
        row=dict(date=day,city=city,hours=len(g),forecast_available_before_utc=issue.isoformat())
        for variable in VARIABLES:
            v=pd.to_numeric(g[variable+'_previous_day1'],errors='coerce')
            row['fc_'+variable]=(float(v.sum()) if variable=='precipitation' else float(v.mean())) if complete and np.isfinite(v).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def weather(start,end,out):
    if end>=pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize():
        raise ValueError('Only completed UTC days may be downloaded; the prospective period is not yet available')
    if start>end:raise ValueError('Start date must precede end date')
    if out.exists():raise ValueError('Output exists; use a new directory')
    cities=pd.read_csv(STATIONS_CSV).groupby('city')[['lat','lon']].mean().reset_index()
    frames=[];raw={}
    for row in cities.itertuples():
        response=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params=dict(latitude=row.lat,longitude=row.lon,
            start_date=start.date().isoformat(),end_date=end.date().isoformat(),timezone='UTC',models='gfs_seamless',
            hourly=','.join(v+'_previous_day1' for v in VARIABLES)),timeout=120)
        response.raise_for_status();data=response.json();frame=aggregate_archive(data,row.city)
        if set(frame.date)!=set(pd.date_range(start,end)):raise ValueError('Archive dates missing: '+row.city)
        frames.append(frame);raw[row.city]=data
    out.mkdir(parents=True)
    for city,data in raw.items():(out/(city+'.json')).write_text(json.dumps(data),encoding='utf-8')
    pd.concat(frames,ignore_index=True).to_csv(out/'daily.csv',index=False)
    manifest=dict(status='quality_checked',dataset_sha256=digest(out/'daily.csv'),lead_hours=24,
        model='gfs_seamless',variables=VARIABLES,start=str(start.date()),end=str(end.date()),
        station_metadata_sha256=digest(STATIONS_CSV),source_sha256={city+'.json':digest(out/(city+'.json')) for city in raw},
        caveats=['City coordinates average configured station locations.','Missing hourly forecasts remain missing; no realized weather substitution.',
                 'Default Open-Meteo units match the original weather model archive.','Original real-time ingestion latency remains unverified.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Checked archived forecast inputs:',out/'daily.csv')

def ground(raw_root,out):
    if out.exists():raise ValueError('Output exists; use a new directory')
    stations=pd.read_csv(STATIONS_CSV);frames=[];audit=[];sources={}
    for row in stations.itertuples():
        relative=Path(row.data_dir).relative_to(Path('data/raw/cpcb'))
        paths=sorted((raw_root/relative).glob('*.xlsx'))
        if not paths:
            audit.append(dict(station_id=row.station_id,city=row.city,files=0,usable_rows=0));continue
        h=pd.concat([parse_cpcb_month_file(p) for p in paths],ignore_index=True)
        h.aqi=pd.to_numeric(h.aqi,errors='raise')
        if ((h.aqi.dropna()<0)|(h.aqi.dropna()>500)).any():raise ValueError('AQI outside valid range: '+row.station_id)
        if h.groupby(['date','hour_ist']).aqi.nunique().gt(1).any():raise ValueError('Conflicting duplicate hourly values: '+row.station_id)
        d=daily_aggregate(h.drop_duplicates(['date','hour_ist']))
        d=d[(d.date>='2026-10-01')&(d.date<'2027-01-01')].copy()
        d['station_id']=row.station_id;d['station_name']=row.station_name;d['city']=row.city
        d['station_lat']=row.lat;d['station_lon']=row.lon;d['aqi_flat_day']=(d.AQI_min==d.AQI_max)&(d.AQI_max<500)
        for col in WEATHER_RAW:d[col]=np.nan
        frames.append(d);sources.update({str(p.resolve()):digest(p) for p in paths})
        audit.append(dict(station_id=row.station_id,city=row.city,files=len(paths),usable_rows=int((d.AQI_eod.notna()&~d.aqi_flat_day).sum())))
    if not frames:raise ValueError('No prospective CPCB files found')
    earlier=[ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv',ROOT/'data/processed/forecast_2026/aligned_dataset.csv']
    histories=[pd.read_csv(p,parse_dates=['date']) for p in earlier]
    histories=[h[h.date<'2026-10-01'] for h in histories]
    result=pd.concat([*histories,*frames],ignore_index=True,sort=False)
    if result.duplicated(['station_id','date']).any():raise ValueError('Duplicate aligned station-days')
    out.mkdir(parents=True);result.to_csv(out/'aligned_dataset.csv',index=False);pd.DataFrame(audit).to_csv(out/'coverage.csv',index=False)
    manifest=dict(status='quality_checked',dataset_sha256=digest(out/'aligned_dataset.csv'),source_sha256=sources,
        historical_sha256={str(p):digest(p) for p in earlier},stations_with_exports=sum(a['files']>0 for a in audit),
        expected_stations=len(stations),available_source_end=str(result.date.max().date()),
        caveats=['At least 12 hourly readings required by daily aggregation; missing endpoints and flat days excluded by model features.',
                 'Unavailable stations remain coverage gaps; no exclusions based on model error.',
                 'December 31 endpoint needs January 2027 hourly exports.','No observed environmental values are fabricated for new dates.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Checked ground inputs:',out/'aligned_dataset.csv')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['ground','weather'])
    ap.add_argument('--raw-root',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--start',default='2026-10-13');ap.add_argument('--end',default='2026-12-31');a=ap.parse_args()
    if a.action=='weather':weather(pd.Timestamp(a.start),pd.Timestamp(a.end),a.output)
    else:
        if not a.raw_root:ap.error('ground requires --raw-root with station subfolders containing monthly CPCB files')
        ground(a.raw_root,a.output)

if __name__=='__main__':main()
