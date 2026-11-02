"""Development-only alerts for high AQI and abrupt changes; never alters frozen forecasts."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def features(d):
    x = d[['city', 'pred', 'aqi_l1']].copy()
    x['pred_change'] = x.pred - x.aqi_l1
    x['abs_pred_change'] = x.pred_change.abs()
    return x


def scores(y, alert):
    tp = int((y & alert).sum())
    hits = int(alert.sum())
    events = int(y.sum())
    return dict(n=len(y), events=events, alerts=hits, true_positives=tp,
                precision=tp/hits if hits else None, recall=tp/events if events else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root, args.out
    dev_path = root/'results/forecast_development/runs/2025-ready-v1/development_predictions.csv.gz'
    test_path = root/'results/forecast_2025/runs/locked-2025-v1/predictions.csv.gz'
    dev = pd.read_csv(dev_path)
    dev = dev[dev.model == 'GRU-28days'].copy()
    test = pd.read_csv(test_path).rename(columns={'reference': 'aqi_l1'})
    assert set(dev.split) == {'selection', 'calibration'}
    assert pd.to_datetime(dev.date).max() < pd.Timestamp('2025-01-01')
    assert pd.to_datetime(test.date).min() >= pd.Timestamp('2025-01-01')
    assert pd.to_datetime(test.date).max() < pd.Timestamp('2026-01-01')
    assert not dev[['pred', 'aqi_l1', 'y']].isna().any().any()
    assert not test[['pred', 'aqi_l1', 'y']].isna().any().any()
    train = dev[dev.split == 'selection']
    calibration = dev[dev.split == 'calibration']
    pre = ColumnTransformer([('numeric', StandardScaler(), ['pred', 'aqi_l1', 'pred_change', 'abs_pred_change']),
                             ('city', OneHotEncoder(handle_unknown='ignore'), ['city'])])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for event in ('severe_aqi', 'sharp_change'):
        def actual(d):
            return (d.y >= 300).to_numpy() if event == 'severe_aqi' else ((d.y-d.aqi_l1).abs() >= 50).to_numpy()
        def baseline(d):
            return (d.pred >= 300).to_numpy() if event == 'severe_aqi' else ((d.pred-d.aqi_l1).abs() >= 50).to_numpy()
        model = make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight='balanced'))
        model.fit(features(train), actual(train))
        p_cal = model.predict_proba(features(calibration))[:, 1]
        base_cal = scores(actual(calibration), baseline(calibration))
        precision_floor = max(0.5, base_cal['precision'] or 0)
        candidates = []
        for threshold in np.linspace(0.01, 0.99, 99):
            m = scores(actual(calibration), p_cal >= threshold)
            if m['precision'] is not None and m['precision'] >= precision_floor:
                candidates.append((m['recall'], m['precision'], threshold))
        threshold = max(candidates)[2] if candidates else None
        candidate_cal = scores(actual(calibration), p_cal >= threshold) if threshold is not None else None
        promotion_eligible = bool(candidate_cal and candidate_cal['recall'] > base_cal['recall'] and
                                  candidate_cal['precision'] >= precision_floor)
        joblib.dump(model, out/f'{event}_candidate.joblib')
        for name, d in [('calibration_2024', calibration), ('historical_test_2025', test)]:
            y = actual(d)
            probs = model.predict_proba(features(d))[:, 1]
            for method, alert in [('frozen_forecast_threshold', baseline(d)),
                                  ('development_alert', probs >= threshold if threshold is not None else np.zeros(len(d), dtype=bool))]:
                results.append(dict(event=event, period=name, method=method,
                                    threshold=threshold if method == 'development_alert' else None,
                                    calibration_precision_floor=precision_floor,
                                    promotion_eligible=promotion_eligible if method == 'development_alert' else False,
                                    average_precision=average_precision_score(y, probs) if method == 'development_alert' else None,
                                    **scores(y, alert)))
    pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
    (out/'sources.json').write_text(json.dumps({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                                 for p in (dev_path, test_path)}, indent=2), encoding='utf-8')
    report = pd.DataFrame(results)
    lines = ['# Development-only event alerts', '',
             'Inputs: frozen GRU forecast, prior-day AQI and city. Classifiers are fit on October 2024 selection rows; '
             'thresholds are chosen on November–December 2024 calibration rows at precision at least 0.50 '
             'and at least the frozen forecast threshold precision there. The January–December 2025 rows are read '
             'only for this historical evaluation. No 2026 target is used. Point AQI forecasts and the prospective GRU default are unchanged.', '',
             'Severe AQI means observed AQI at least 300. Sharp change means absolute change from previous day at least 50 AQI.', '',
             '| Event | Period | Method | Events | Alerts | Precision | Recall |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in report.itertuples():
        precision = '—' if pd.isna(r.precision) else f'{r.precision:.3f}'
        recall = '—' if pd.isna(r.recall) else f'{r.recall:.3f}'
        lines.append(f'| {r.event} | {r.period} | {r.method} | {r.events} | {r.alerts} | '
                     f'{precision} | {recall} |')
    lines += ['', 'Neither candidate passes the calibration promotion rule when promotion_eligible is false. '
              'A precision gain at the expense of recall is not evidence of better event detection. '
              'This is an alerting experiment, not an improvement to numerical AQI forecast error. '
              'No candidate is deployed; future work needs richer forecast-safe inputs and independent evaluation.', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
