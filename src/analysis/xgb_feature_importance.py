"""Read-only diagnostic: feature importance of the frozen weather XGBoost (2026-ready-v1).

Three views, each grouped by what the inputs mean because the 28 daily lags are
highly correlated: the model's own gain, mean absolute tree-SHAP contribution and
the RMSE rise when a group is shuffled. Output goes to results/analysis/.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/models'))
from forecast_development import feature_table, WEATHER_RAW

DEV = ROOT / 'results/forecast_development/runs/2026-ready-v1'
OUT = ROOT / 'results/analysis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((DEV / 'manifest.json').read_text())
    assert sha256((DEV / 'forecast_model.joblib').read_bytes()).hexdigest() == frozen['artifact_sha256']
    art = joblib.load(DEV / 'forecast_model.joblib')
    model, feats = art['model'], art['features']
    model.set_params(device='cpu')

    weather = pd.read_csv(ROOT / 'data/processed/forecast_weather/daily.csv', parse_dates=['date', 'forecast_available_before_utc'])
    hist = pd.read_csv(ROOT / 'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv', parse_dates=['date'])
    fresh = pd.read_csv(ROOT / 'data/processed/forecast_2026/aligned_dataset.csv', parse_dates=['date'])
    aligned = pd.concat([hist, fresh], ignore_index=True)
    for col in WEATHER_RAW:
        if col not in aligned: aligned[col] = np.nan
    df, _ = feature_table(aligned)
    df = df.merge(weather, on=['city', 'date'], how='left', validate='many_to_one')
    dev = df[(df.date >= '2025-10-01') & (df.date < '2026-01-01')].reset_index(drop=True)
    test = df[(df.date >= '2026-01-01') & (df.date < '2026-10-01')].reset_index(drop=True)

    pred = lambda d: np.clip(d.aqi_l1.to_numpy() + model.predict(d[feats]), 0, 500)
    saved = pd.read_csv(ROOT / 'results/forecast_2026/runs/locked-2026-v1/predictions.csv.gz', parse_dates=['date'])
    match = saved.set_index(['date', 'station_id']).pred.reindex(pd.MultiIndex.from_arrays([test.date, test.station_id]))
    keep = match.notna().to_numpy()
    replication = float(np.abs(pred(test)[keep] - match.to_numpy()[keep]).max())
    test = test[keep].reset_index(drop=True)

    groups = {
        "Yesterday's AQI": ['aqi_l1'],
        'Older daily AQI (days 2-28)': [f'aqi_l{k}' for k in range(2, 29)],
        'AQI summaries': [f'aqi_{s}{k}' for k in (3, 7, 14, 28) for s in ('mean', 'sd', 'count')],
        'Day-to-day change': ['aqi_change1'],
        'Day of year': ['doy_sin', 'doy_cos'],
        'Station location': ['lat', 'lon'],
        'Weather forecast': [c for c in feats if c.startswith('fc_')],
    }
    assert sorted(sum(groups.values(), [])) == sorted(feats)
    fc = groups['Weather forecast']
    booster = model.get_booster()
    gain = pd.Series(booster.get_score(importance_type='total_gain')).reindex(feats).fillna(0)
    gain_share = {g: float(gain[c].sum() / gain.sum()) for g, c in groups.items()}

    rmse = lambda d, p: float(np.sqrt(np.mean((p - d.y.to_numpy()) ** 2)))
    rng = np.random.default_rng(0)

    def permute(d, cols, within_city=False):
        out = []
        for _ in range(5):
            x = d.copy()
            if within_city:
                for _, idx in d.groupby('city').indices.items():
                    x.iloc[idx, [x.columns.get_loc(c) for c in cols]] = d.iloc[rng.permutation(idx)][cols].to_numpy()
            else:
                x[cols] = d[cols].iloc[rng.permutation(len(d))].to_numpy()
            out.append(rmse(d, pred(x)))
        return float(np.mean(out))

    result = dict(selected=art['selected'], features=len(feats), replication_max_diff=replication, gain_share=gain_share)
    for name, d in (('development', dev), ('test_2026', test)):
        base, ref = rmse(d, pred(d)), rmse(d, d.aqi_l1.to_numpy())
        contrib = booster.predict(xgb.DMatrix(d[feats]), pred_contribs=True)[:, :-1]
        per_feature = pd.Series(np.abs(contrib).mean(axis=0), index=feats)
        miss = d.copy(); miss[fc] = np.nan
        entry = dict(rows=len(d), model_rmse=base, baseline_rmse=ref,
                     groups={g: dict(shap_share=float(per_feature[c].sum() / per_feature.sum()), rmse_rise=permute(d, c) - base) for g, c in groups.items()},
                     weather_within_city_rise=permute(d, fc, within_city=True) - base, weather_missing_rmse=rmse(miss, pred(miss)),
                     weather_variables={c[3:]: dict(shap_share=float(per_feature[c] / per_feature.sum()), rmse_rise=permute(d, [c]) - base) for c in fc})
        if name == 'test_2026':
            start = time.perf_counter(); pred(d); seconds = time.perf_counter() - start
            entry['inference'] = dict(rows=len(d), seconds=seconds, ms_per_1000=seconds / len(d) * 1e6, device='cpu')
        result[name] = entry
    (OUT / 'xgb_feature_importance.json').write_text(json.dumps(result, indent=1))
    print('saved', OUT / 'xgb_feature_importance.json', 'replication diff', replication)


if __name__ == '__main__':
    main()
