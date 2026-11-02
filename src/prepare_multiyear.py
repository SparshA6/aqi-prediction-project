"""Checked, resumable 2024 ingestion; the verified 2023 table is preserved."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from combine_with_aqi import daily_aggregate, parse_cpcb_month_file, nearest_grid_point, aqi_band, FILENAME_RE, MONTH_NAMES
from paths import ROOT, STATIONS_CSV, CITIES_CSV
OUT = ROOT / 'data/processed/multiyear_2023_2024'
ORIGINAL = ROOT / 'data/processed/aligned_dataset.csv'
def digest(p): return sha256(p.read_bytes()).hexdigest()
def station_files(row):
    relative = Path(row.data_dir).relative_to(Path('data/raw/cpcb'))
    return sorted((ROOT / 'data/raw/cpcb_2024' / relative).glob('*2024*.xlsx'))
def readiness():
    missing = []
    for city in pd.read_csv(CITIES_CSV).city:
        p = ROOT / 'data/raw/gee_2024' / f'{city}.csv'
        if not p.exists() or set(pd.to_datetime(pd.read_csv(p, usecols=['date']).date)) != set(pd.date_range('2024-01-01','2024-12-31')):
            missing.append(f'{city}: incomplete grid dates')
    for row in pd.read_csv(STATIONS_CSV).itertuples():
        months = set()
        for p in station_files(row):
            m = FILENAME_RE.search(p.name)
            if m and (m.group('year_first') or m.group('year_second')) == '2024': months.add((m.group('month_a') or m.group('month_b')).lower())
        if months != {m.lower() for m in MONTH_NAMES}: missing.append(f'{row.station_id}: {len(months)}/12 distinct months')
    return missing

def prepare(resume=False):
    if OUT.exists() and not resume: raise SystemExit('Staging folder exists; use --resume')
    OUT.mkdir(parents=True,exist_ok=True)
    cache, grids = OUT/'station_cache', OUT/'gee'
    cache.mkdir(exist_ok=True); grids.mkdir(exist_ok=True)
    original_hash = digest(ORIGINAL)
    old = pd.read_csv(ORIGINAL,parse_dates=['date'])
    if set(old.date.dt.year) != {2023} or old.duplicated(['station_id','date']).any(): raise ValueError('Invalid original 2023 table')
    stations = pd.read_csv(STATIONS_CSV)
    frames, audit = [old], []
    sources = {str(ORIGINAL.relative_to(ROOT)):original_hash,str(STATIONS_CSV.relative_to(ROOT)):digest(STATIONS_CSV)}
    count = 0
    for city in pd.read_csv(CITIES_CSV).city:
        paths = [ROOT/'data/raw/gee'/f'{city}.csv',ROOT/'data/raw/gee_2024'/f'{city}.csv']
        grid = pd.concat([pd.read_csv(p,parse_dates=['date']) for p in paths],ignore_index=True)
        sources.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
        counts = grid.groupby('date').size()
        if grid.duplicated(['date','latitude','longitude']).any() or counts.nunique()!=1 or len(grid[['latitude','longitude']].drop_duplicates())!=counts.iloc[0] or set(counts.index)!=set(pd.date_range('2023-01-01','2024-12-31')): raise ValueError(f'Invalid grid geometry/dates: {city}')
        grid.to_csv(grids/f'{city}.csv',index=False)
        g24 = grid[grid.date.dt.year==2024]
        sub = stations[stations.city==city]
        matched = sub.merge(nearest_grid_point(sub,g24),on='station_id')
        for _, row in matched.iterrows():
            files = station_files(row)
            hashes = {str(p.relative_to(ROOT)):digest(p) for p in files}
            sources.update(hashes)
            fp = sha256(json.dumps([hashes,digest(Path(__file__)),row.to_dict()],sort_keys=True,default=str).encode()).hexdigest()
            cp, mp = cache/f'{row.station_id}.csv', cache/f'{row.station_id}.json'
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            if cp.exists() and meta.get('fingerprint')==fp and meta.get('csv_sha256')==digest(cp):
                merged = pd.read_csv(cp,parse_dates=['date']); details = meta['audit']
            else:
                hourly = pd.concat([parse_cpcb_month_file(p) for p in files],ignore_index=True)
                hourly['aqi'] = pd.to_numeric(hourly.aqi,errors='raise')
                if ((hourly.aqi.dropna()<0)|(hourly.aqi.dropna()>500)).any(): raise ValueError(f'Out-of-range AQI: {row.station_id}')
                dup = hourly.duplicated(['date','hour_ist'],keep=False)
                if hourly[dup].groupby(['date','hour_ist']).aqi.nunique().gt(1).any(): raise ValueError(f'Conflicting hourly duplicates: {row.station_id}')
                daily = daily_aggregate(hourly.drop_duplicates(['date','hour_ist']))
                daily = daily[daily.date.dt.year==2024]
                cell = g24[np.isclose(g24.latitude,row.grid_lat)&np.isclose(g24.longitude,row.grid_lon)]
                merged = daily.merge(cell,on='date',how='inner',validate='one_to_one')
                for col, source in [('station_id','station_id'),('station_name','station_name'),('station_lat','lat'),('station_lon','lon'),('grid_lat','grid_lat'),('grid_lon','grid_lon'),('match_distance_km','match_distance_km')]: merged[col]=row[source]
                merged['AQI_band']=merged.AQI_eod.map(aqi_band)
                merged['AQI_mean_band']=merged.AQI_mean.map(aqi_band)
                merged['aqi_flat_day']=(merged.AQI_min==merged.AQI_max)&(merged.AQI_max<500)
                merged['satellite_bands_present']=merged[['NO2','HCHO','AOD']].notna().sum(axis=1)
                details=dict(station_id=row.station_id,city=city,files=len(files),duplicate_hourly_rows=int(dup.sum()),rows=len(merged),missing_target=int(merged.AQI_eod.isna().sum()),flat_days=int(merged.aqi_flat_day.sum()))
                merged.to_csv(cp,index=False)
                mp.write_text(json.dumps(dict(fingerprint=fp,csv_sha256=digest(cp),audit=details),indent=2))
            frames.append(merged); audit.append(details); count+=1
            print(f'Checked {count}/{len(stations)}: {city} / {row.station_id} ({len(merged)} daily rows)',flush=True)
    result=pd.concat(frames,ignore_index=True)[old.columns]
    if result.duplicated(['station_id','date']).any(): raise ValueError('Duplicate station/day keys')
    pd.testing.assert_frame_equal(result[result.date.dt.year==2023].reset_index(drop=True),old,check_dtype=False)
    if digest(ORIGINAL)!=original_hash: raise ValueError('Protected original changed')
    usable=result[result.AQI_eod.notna()&~result.aqi_flat_day].copy(); usable['year']=usable.date.dt.year
    coverage=usable.groupby(['city','year']).agg(station_days=('station_id','size'),stations=('station_id','nunique'))
    if {(c,y) for c in stations.city.unique() for y in (2023,2024)}-set(coverage.index): raise ValueError('Missing usable city-year data')
    coverage.to_csv(OUT/'coverage_by_city_year.csv')
    pd.DataFrame(audit).to_csv(OUT/'station_ingestion_audit.csv',index=False)
    temp=OUT/'aligned_dataset.tmp.csv'; result.sort_values(['date','station_id']).to_csv(temp,index=False); temp.replace(OUT/'aligned_dataset.csv')
    manifest=dict(status='quality_checked',years=[2023,2024],rows=len(result),usable_rows=len(usable),stations=int(result.station_id.nunique()),cities=sorted(result.city.unique()),dataset_sha256=digest(OUT/'aligned_dataset.csv'),original_2023_sha256=original_hash,original_2023_preserved=True,source_sha256=sources,staging_code_sha256=digest(Path(__file__)),caveats=['At least 12 valid hourly readings required; missing targets and stuck-sensor days excluded from models.','December 31 2024 EOD requires January 2025 input, not collected.','Fixed station metadata; relocations and sensor changes not independently verified.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(coverage.to_string(),flush=True); print('Checked multi-year dataset saved',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--resume',action='store_true'); ap.add_argument('--wait',action='store_true'); a=ap.parse_args()
    import time
    while True:
        missing=readiness()
        if not missing: break
        if not a.wait: raise SystemExit('Not ready:\n'+'\n'.join(missing))
        print(f'Waiting for {len(missing)} inputs',flush=True); time.sleep(60)
    prepare(a.resume)
