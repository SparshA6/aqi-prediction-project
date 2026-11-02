"""2025-ready forecast inputs. Only earlier calendar observations enter features."""
import numpy as np
import pandas as pd

WEATHER_RAW = ['temperature_2m','dewpoint_temperature_2m','u_component_of_wind_10m',
               'v_component_of_wind_10m','surface_pressure','total_precipitation_sum']

def development_splits(dates):
    d = pd.to_datetime(dates)
    masks = dict(train=(d >= '2023-01-01') & (d < '2024-10-01'),
                 selection=(d >= '2024-10-01') & (d < '2024-11-01'),
                 calibration=(d >= '2024-11-01') & (d < '2025-01-01'),
                 test=(d >= '2025-01-01') & (d < '2026-01-01'))
    return {k: np.flatnonzero(np.asarray(v)) for k,v in masks.items()}

def feature_table(aligned):
    a=aligned.copy(); a['date']=pd.to_datetime(a.date)
    if a.duplicated(['station_id','date']).any(): raise ValueError('Duplicate station/date')
    valid=a[a.AQI_eod.notna() & ~a.aqi_flat_day].copy()
    out=valid[['date','station_id','city','station_name','station_lat','station_lon']].copy()
    out['y']=valid.AQI_eod
    out['lat']=out.station_lat; out['lon']=out.station_lon
    lookup=valid.set_index(['station_id','date']).AQI_eod
    history=[]
    for lag in range(1,29):
        key=pd.MultiIndex.from_arrays([out.station_id,out.date-pd.Timedelta(days=lag)])
        values=lookup.reindex(key).to_numpy(); out[f'aqi_l{lag}']=values; history.append(values)
    matrix=np.column_stack(history)
    for days in (3,7,14,28):
        h=matrix[:,:days]; count=np.isfinite(h).sum(axis=1)
        mean=np.divide(np.nansum(h,axis=1),count,out=np.full(len(h),np.nan),where=count>0)
        variance=np.divide(np.nansum((h-mean[:,None])**2,axis=1),count,out=np.full(len(h),np.nan),where=count>0)
        out[f'aqi_mean{days}']=mean; out[f'aqi_sd{days}']=np.sqrt(variance); out[f'aqi_count{days}']=count
    out['aqi_change1']=out.aqi_l1-out.aqi_l2
    phase=2*np.pi*out.date.dt.dayofyear/365.25
    out['doy_sin']=np.sin(phase); out['doy_cos']=np.cos(phase)
    # Seven-day minimum lag allows the usual five-day ERA5-Land release delay
    # plus a margin. Historical final reanalysis is not a publication-vintage archive.
    weather=a.set_index(['station_id','date'])[WEATHER_RAW]
    for lag in (7,14):
        key=pd.MultiIndex.from_arrays([out.station_id,out.date-pd.Timedelta(days=lag)])
        w=weather.reindex(key)
        for col in WEATHER_RAW: out[f'{col}_l{lag}']=w[col].to_numpy()
    out=out[out.aqi_l1.notna()].sort_values(['date','station_id']).reset_index(drop=True)
    base=['aqi_l1','aqi_l2','aqi_l3','aqi_l8','aqi_mean3','doy_sin','doy_cos','lat','lon']
    long=[f'aqi_l{k}' for k in range(1,29)]+[f'aqi_{stat}{k}' for k in (3,7,14,28) for stat in ('mean','sd','count')]+['aqi_change1','doy_sin','doy_cos','lat','lon']
    return out,dict(short=base,long=long,delayed_weather=long+[f'{c}_l{k}' for k in (7,14) for c in WEATHER_RAW])
