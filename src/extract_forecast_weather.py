"""Archive 24-hour-ahead GFS weather, at public station-coordinate city means."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from paths import ROOT,STATIONS_CSV
VARIABLES=['temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation','surface_pressure']
OUT=ROOT/'data/processed/forecast_weather'

def fetch(row):
    path=OUT/f'{row.city}.json'
    if not path.exists():
        params=dict(latitude=row.lat,longitude=row.lon,start_date='2024-01-01',end_date='2026-09-30',timezone='UTC',models='gfs_seamless',hourly=','.join(v+'_previous_day1' for v in VARIABLES))
        for attempt in range(4):
            try:
                response=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params=params,timeout=120)
                response.raise_for_status(); data=response.json(); break
            except requests.RequestException:
                if attempt==3: raise
                time.sleep(5)
        path.write_text(json.dumps(data),encoding='utf-8')
    data=json.loads(path.read_text()); hourly=pd.DataFrame(data['hourly']); hourly['date']=pd.to_datetime(hourly.time).dt.floor('D')
    rows=[]
    for date,g in hourly.groupby('date'):
        result=dict(date=date,city=row.city,hours=len(g),forecast_available_before_utc=date,
                    forecast_lat=data['latitude'],forecast_lon=data['longitude'])
        for var in VARIABLES:
            values=pd.to_numeric(g[var+'_previous_day1'],errors='coerce')
            result['fc_'+var]=float(values.sum()) if var=='precipitation' and values.notna().sum()==24 else float(values.mean()) if values.notna().sum()==24 else np.nan
        rows.append(result)
    print(f'{row.city}: {len(rows)} archived forecast days',flush=True)
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cities=pd.read_csv(STATIONS_CSV).groupby('city')[['lat','lon']].mean().reset_index()
    with ThreadPoolExecutor(max_workers=4) as pool: frames=list(pool.map(fetch,cities.itertuples()))
    df=pd.concat(frames).sort_values(['city','date']); df.to_csv(OUT/'daily.csv',index=False)
    manifest=dict(status='complete',provider='Open-Meteo Previous Runs API, NCEP GFS Seamless',retrieved_utc=datetime.now(timezone.utc).isoformat(),lead_hours=24,
        source='https://open-meteo.com/en/docs/previous-runs-api',model='gfs_seamless',station_coordinates_sha256=sha256(STATIONS_CSV.read_bytes()).hexdigest(),
        dataset_sha256=sha256((OUT/'daily.csv').read_bytes()).hexdigest(),raw_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.json') if p.name!='manifest.json'},
        caveats=['City mean coordinates shared across stations; weather is coarser than station-scale pollution.',
                 'Each valid hour uses the forecast made 24 hours earlier; all are before UTC target-day issue time.',
                 'Archive issue offset does not verify original provider ingestion latency or later corrections.',
                 'Unavailable hours remain missing; final reanalysis is never substituted.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
