"""Calendar splits and forecast windows that never use future observations."""
import numpy as np
import pandas as pd

def split_years(dates):
    d = pd.to_datetime(dates)
    masks = dict(train=(d >= '2023-01-01') & (d < '2023-10-01'),
                 selection=(d >= '2023-10-01') & (d < '2023-11-01'),
                 calibration=(d >= '2023-11-01') & (d < '2024-01-01'),
                 test=(d >= '2024-01-01') & (d < '2025-01-01'))
    return {k: np.flatnonzero(np.asarray(v)) for k, v in masks.items()}

def forecast_windows(aligned, rows, window=7):
    a = aligned.copy()
    a['date'] = pd.to_datetime(a.date)
    a = a[a.AQI_eod.notna() & ~a.aqi_flat_day]
    if a.duplicated(['station_id', 'date']).any(): raise ValueError('Duplicate forecast history keys')
    lookup = a.set_index(['station_id', 'date']).AQI_eod
    result = []
    for lag in range(window, 0, -1):
        dates = pd.to_datetime(rows.date) - pd.Timedelta(days=lag)
        key = pd.MultiIndex.from_arrays([rows.station_id, dates])
        history = lookup.reindex(key).to_numpy()
        phase = 2*np.pi*dates.dt.dayofyear.to_numpy()/365.25
        result.append(np.column_stack([history, np.isfinite(history).astype(float), np.sin(phase), np.cos(phase), rows.lat, rows.lon]))
    return np.stack(result, axis=1).astype(np.float32)

def normalize_windows(train_fit, values):
    mu = np.nanmean(train_fit, axis=(0, 1))
    sd = np.nanstd(train_fit, axis=(0, 1))
    mu = np.where(np.isfinite(mu), mu, 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1)
    return np.nan_to_num((values-mu)/sd).astype(np.float32), mu, sd

def paired_week_bootstrap(rows, repetitions=2000):
    """Paired seven-day block bootstrap, keeping all stations on each date together."""
    data = rows.copy()
    data['date'] = pd.to_datetime(data.date)
    data['se_selected'] = (data.pred-data.y)**2
    data['se_reference'] = (data.reference-data.y)**2
    daily = data.groupby('date').agg(selected=('se_selected','sum'), reference=('se_reference','sum'), n=('y','size'))
    calendar = pd.date_range('2024-01-01','2024-12-31')
    daily = daily.reindex(calendar, fill_value=0)
    blocks = daily.groupby(np.arange(len(calendar))//7).sum().to_numpy()
    blocks = blocks[blocks[:,2] > 0]
    if not len(blocks): raise ValueError('No scored 2024 observations for bootstrap')
    rng = np.random.default_rng(0)
    draws = blocks[rng.integers(0, len(blocks), size=(repetitions, len(blocks)))].sum(axis=1)
    gains = np.sqrt(draws[:,1]/draws[:,2]) - np.sqrt(draws[:,0]/draws[:,2])
    return dict(gain=float(np.sqrt(daily.reference.sum()/daily.n.sum())-np.sqrt(daily.selected.sum()/daily.n.sum())),
                lower=float(np.quantile(gains,.025)), upper=float(np.quantile(gains,.975)),
                caveat='Seven-day block bootstrap; residual longer-range dependence and station changes remain possible.')
