"""Read-only diagnostic: how much does the frozen 2025 GRU use beyond yesterday's AQI?

Re-runs the frozen GRU on every 2025 test row with parts of its input removed and
compares it with simple linear models fitted on the same training period. Nothing
in results/forecast_* is modified; output goes to results/analysis/.
"""
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/models'))
from forecast_development import feature_table
from year_holdout import forecast_windows
from seq_models import make_net, _predict

DEV = ROOT / 'results/forecast_development/runs/2025-ready-v1'
OUT = ROOT / 'results/analysis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    art = joblib.load(DEV / 'forecast_model.joblib')
    aligned = pd.read_csv(ROOT / 'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv', parse_dates=['date'])
    df, _ = feature_table(aligned)
    test = df[(df.date >= '2025-01-01') & (df.date < '2026-01-01')].reset_index(drop=True)
    train = df[(df.date >= '2023-01-01') & (df.date < '2024-10-01')].reset_index(drop=True)
    saved = pd.read_csv(ROOT / 'results/forecast_2025/runs/locked-2025-v1/predictions.csv.gz', parse_dates=['date'])

    raw = forecast_windows(aligned, test, art['window'])  # (n, 28, 6): AQI, valid, sin, cos, lat, lon
    mu, sd = art['mu'], art['sd']
    nets = []
    for ck in art['checkpoints']:
        net = make_net('GRU', art['n_in'])
        net.load_state_dict(torch.load(DEV / 'models' / ck['weight'], map_location='cpu', weights_only=True))
        nets.append((net, ck))

    def run(window):
        x = np.nan_to_num((window - mu) / sd).astype(np.float32)
        return np.mean([np.clip(_predict(n, x) * c['target_std'] + c['target_mean'], 0, 500) for n, c in nets], axis=0)

    y, ref = test.y.to_numpy(), test.aqi_l1.to_numpy()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    mae = lambda p: float(np.mean(np.abs(p - y)))

    start = time.perf_counter(); full = run(raw); seconds = time.perf_counter() - start
    key = pd.MultiIndex.from_arrays([test.date, test.station_id])
    check = saved.set_index(['date', 'station_id']).pred.reindex(key).to_numpy()
    replication = float(np.nanmax(np.abs(full - check)))

    latest = raw.copy(); latest[:, :-1, 0] = raw[:, -1:, 0]
    nocal = raw.copy(); nocal[:, :, 2:6] = mu[2:6]
    only = latest.copy(); only[:, :, 2:6] = mu[2:6]
    shuf = raw.copy(); rng = np.random.default_rng(0)
    for i in range(len(shuf)):
        shuf[i, :-1, 0] = shuf[i, rng.permutation(27), 0]
    variants = {'full': full, 'yesterday_repeated': run(latest), 'no_calendar_location': run(nocal),
                'yesterday_only': run(only), 'older_days_shuffled': run(shuf)}
    gain_full = rmse(ref) - rmse(full)
    table = {name: dict(rmse=rmse(p), mae=mae(p), gain=rmse(ref) - rmse(p), share_of_gain=(rmse(ref) - rmse(p)) / gain_full,
                        corr_with_yesterday=float(np.corrcoef(p, ref)[0, 1]), mean_abs_gap=float(np.mean(np.abs(p - ref))))
             for name, p in variants.items()}

    def linear(cols, model):
        fill = train[cols].mean()
        model.fit(train[cols].fillna(fill), train.y)
        p = np.clip(model.predict(test[cols].fillna(fill)), 0, 500)
        return dict(rmse=rmse(p), gain=rmse(ref) - rmse(p), share_of_gain=(rmse(ref) - rmse(p)) / gain_full)
    history = [c for c in test.columns if c.startswith('aqi_') and not c.startswith('aqi_count')] + ['doy_sin', 'doy_cos', 'lat', 'lon']
    linear_models = {'yesterday_only': linear(['aqi_l1'], LinearRegression()),
                     'three_recent': linear(['aqi_l1', 'aqi_l2', 'aqi_mean7'], Ridge(1.0)),
                     'all_history': linear(history, Ridge(10.0))}
    fit = LinearRegression().fit(train[['aqi_l1']], train.y)
    d, err = full - ref, y - ref
    departure = dict(within5=float(np.mean(np.abs(d) <= 5)), within10=float(np.mean(np.abs(d) <= 10)), within20=float(np.mean(np.abs(d) <= 20)),
                     std_model_change=float(d.std()), std_actual_change=float(err.std()),
                     change_correlation=float(np.corrcoef(d, err)[0, 1]), change_slope=float(np.polyfit(d, err, 1)[0]),
                     one_variable_intercept=float(fit.intercept_), one_variable_slope=float(fit.coef_[0]))
    result = dict(rows=len(test), baseline_rmse=rmse(ref), baseline_mae=mae(ref), replication_max_diff=replication,
                  variants=table, linear=linear_models, departure=departure,
                  inference=dict(rows=len(test), seconds=seconds, ms_per_1000=seconds / len(test) * 1e6, device='cpu', networks=len(nets)))
    (OUT / 'gru_input_ablation.json').write_text(json.dumps(result, indent=1))
    print('saved', OUT / 'gru_input_ablation.json', 'replication diff', round(replication, 4))


if __name__ == '__main__':
    main()
