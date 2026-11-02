"""Development-only blend selection, frozen prospective inference and evaluation.

Previously inspected 2026 targets are used only in the separate error report.
No ensemble weights are selected or assessed on those targets.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/models'))
from forecast_diagnostics import summarize
from evaluate_forecast_2025 import predict, scores
from forecast_development import feature_table

RUN = ROOT / 'results/ensemble/runs/prospective-oct13-v1'
GRU = ROOT / 'results/forecast_development/runs/2025-ready-v1'
WEATHER = ROOT / 'results/forecast_development/runs/2026-ready-v1'
KEYS = ['date', 'station_id', 'city', 'split']

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def blend(gru, weather, weight):
    if not 0 <= weight <= 1:
        raise ValueError('Weather weight must be between zero and one')
    gru, weather = np.asarray(gru), np.asarray(weather)
    if gru.shape != weather.shape or not np.isfinite(gru).all() or not np.isfinite(weather).all():
        raise ValueError('Both components must have matching finite predictions')
    return np.clip((1-weight)*gru + weight*weather, 0, 500)

def paired_development(frame):
    """One-to-one key join, rather than row-order-dependent blending."""
    a = frame[frame.model == 'Frozen GRU-28days'].copy()
    b = frame[frame.model == 'XGBoost-change-forecast-weather-d5'].copy()
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
        raise ValueError('Duplicate component keys')
    paired = a.merge(b, on=KEYS, suffixes=('_gru', '_weather'), validate='one_to_one', how='outer', indicator=True)
    if not (paired['_merge'] == 'both').all():
        raise ValueError('Component cohorts differ')
    if not np.array_equal(paired.y_gru, paired.y_weather) or not np.array_equal(paired.aqi_l1_gru, paired.aqi_l1_weather):
        raise ValueError('Component targets or references disagree')
    paired['y'] = paired.y_gru
    paired['reference'] = paired.aqi_l1_gru
    return paired

def select_weight(paired):
    selection = paired[paired.split == 'selection'].copy()
    if selection.empty or not selection.date.astype(str).str.startswith('2025-10-').all():
        raise ValueError('Selection must contain October 2025 only')
    base = summarize(selection.assign(pred=selection.pred_gru))
    candidates = []
    for w in np.linspace(0, 1, 21):
        s = summarize(selection.assign(pred=blend(selection.pred_gru, selection.pred_weather, w)))
        eligible = all(base[k] is not None and s[k] is not None and s[k] >= base[k] for k in ['precision', 'recall'])
        candidates.append(dict(weather_weight=float(w), eligible=eligible, **s))
    table = pd.DataFrame(candidates)
    valid = table[table.eligible]
    if valid.empty:
        raise ValueError('No eligible candidate; do not relax constraints after selection')
    winner = valid.sort_values(['RMSE', 'weather_weight']).iloc[0]
    return float(winner.weather_weight), table

def develop():
    if RUN.exists():
        raise ValueError('Frozen run already exists; use a new version instead of overwriting')
    paths = [WEATHER/'development_predictions.csv.gz', WEATHER/'forecast_model.joblib', GRU/'forecast_model.joblib']
    paired = paired_development(pd.read_csv(paths[0]))
    if set(paired.split) != {'selection', 'calibration'} or (pd.to_datetime(paired.date).dt.year >= 2026).any():
        raise ValueError('Development targets must precede 2026')
    weight, grid = select_weight(paired)
    cal = paired[paired.split == 'calibration'].copy()
    if cal.empty or not ((pd.to_datetime(cal.date) >= '2025-11-01') & (pd.to_datetime(cal.date) < '2026-01-01')).all():
        raise ValueError('Calibration must be November-December 2025')
    cal['pred'] = blend(cal.pred_gru, cal.pred_weather, weight)
    radius = float(np.quantile(abs(cal.y-cal.pred), min(1, np.ceil((len(cal)+1)*.9)/len(cal)), method='higher'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    g = joblib.load(GRU/'forecast_model.joblib')
    for ck in g['checkpoints']:
        path = GRU/'models'/ck['weight']; hashes[str(path.relative_to(ROOT))] = digest(path)
    for name in ['ensemble_followthrough.py','forecast_development.py','evaluate_forecast_2025.py','year_holdout.py','seq_models.py']:
        path = ROOT/'src/models'/name; hashes[str(path.relative_to(ROOT))] = digest(path)
    manifest = dict(status='frozen_awaiting_prospective_data', frozen_utc=datetime.now(timezone.utc).isoformat(),
        weather_weight=weight, gru_weight=1-weight, interval_radius=radius, input_sha256=hashes,
        selection_period=['2025-10-01','2025-10-31'], calibration_period=['2025-11-01','2025-12-31'],
        prospective_period=['2026-10-13','2026-12-31'], earliest_final_evaluation_utc='2027-01-01',
        selection_rule='Lowest October 2025 RMSE among weights 0,0.05,...,1 with no precision or recall loss versus GRU at 50 AQI threshold; smaller weather weight breaks ties',
        promotion_rule='95% paired weekly RMSE-gain lower bound >0 against both GRU and persistence, with no sharp-change precision or recall loss against GRU',
        tested_on_fresh_data=False, default_model_changed=False,
        limitations=['2025 has been examined in earlier studies; development scores are not independent evidence.',
                    'January-September 2026 targets are already seen and must not be used to select this blend.',
                    'October 13-December 31 targets are prospective as of October 12; actual export availability remains unknown.',
                    'Intervals use historical calibration; temporal dependence and drift may affect coverage.',
                    'Archived hourly forecasts precede each target day; historical ingestion latency is unverified.'])
    RUN.mkdir(parents=True)
    grid.to_csv(RUN/'selection_grid.csv', index=False)
    cal.to_csv(RUN/'calibration_predictions.csv.gz', index=False)
    (RUN/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

def load_frozen():
    m = json.loads((RUN/'manifest.json').read_text())
    for rel, expected in m['input_sha256'].items():
        if digest(ROOT/rel) != expected:
            raise ValueError('Frozen input or inference code changed: '+rel)
    return m

def prospective_ready(manifest, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None: now = now.tz_localize('UTC')
    return now >= pd.Timestamp(manifest['earliest_final_evaluation_utc'], tz='UTC')

def evaluate(aligned_path, weather_path):
    m = load_frozen()
    if not prospective_ready(m):
        raise ValueError('Fresh period has not ended. Final evaluation opens on 1 January 2027 UTC.')
    for path in [aligned_path, weather_path]:
        provenance = json.loads(path.with_name('manifest.json').read_text())
        if provenance.get('status') != 'quality_checked' or provenance.get('dataset_sha256') != digest(path):
            raise ValueError('Require matching quality_checked manifest and dataset hash: '+str(path))
    aligned = pd.read_csv(aligned_path, parse_dates=['date'])
    if aligned.duplicated(['station_id','date']).any(): raise ValueError('Duplicate station dates')
    if aligned.date.max() < pd.Timestamp(m['prospective_period'][1]): raise ValueError('Final target date absent')
    rows, _ = feature_table(aligned)
    rows = rows[rows.date.between(*m['prospective_period'])].copy()
    weather = pd.read_csv(weather_path, parse_dates=['date','forecast_available_before_utc'])
    fc = joblib.load(WEATHER/'forecast_model.joblib')
    weather_cols = [c for c in fc['features'] if c.startswith('fc_')]
    valid_issue = pd.to_datetime(weather.forecast_available_before_utc, utc=True)
    if (valid_issue >= pd.to_datetime(weather.date, utc=True)).any(): raise ValueError('Forecast issue time must precede target day')
    rows = rows.merge(weather[['date','city',*weather_cols]], on=['date','city'], how='left', validate='many_to_one')
    before = len(rows)
    rows = rows.dropna(subset=weather_cols)
    if len(rows) < 100 or rows.date.nunique() < 30: raise ValueError('Insufficient paired future cohort')
    g = predict(joblib.load(GRU/'forecast_model.joblib'), GRU, aligned, rows, 'cpu')
    w = predict(fc, WEATHER, aligned, rows, 'cpu')
    result = rows[['date','station_id','city','y','aqi_l1']].rename(columns={'aqi_l1':'reference'}).copy()
    result['gru_pred'] = g; result['weather_pred'] = w
    result['pred'] = blend(g, w, m['weather_weight'])
    result['lower'] = np.maximum(0,result.pred-m['interval_radius'])
    result['upper'] = np.minimum(500,result.pred+m['interval_radius'])
    metrics=[]
    for name,col in [('Ensemble','pred'),('Frozen GRU','gru_pred'),('Weather','weather_pred'),('Persistence','reference')]:
        metrics.append(dict(model=name, **summarize(result.assign(pred=result[col]))))
    gain={}
    days=pd.date_range(*m['prospective_period'])
    for name,col in [('gru','gru_pred'),('persistence','reference')]:
        r=result.assign(a=(result.pred-result.y)**2,b=(result[col]-result.y)**2)
        daily=r.groupby('date').agg(a=('a','sum'),b=('b','sum'),n=('y','size')).reindex(days,fill_value=0)
        blocks=daily.groupby(np.arange(len(days))//7).sum().to_numpy();blocks=blocks[blocks[:,2]>0]
        samples=blocks[np.random.default_rng(42).integers(0,len(blocks),(2000,len(blocks)))].sum(axis=1)
        dif=np.sqrt(samples[:,1]/samples[:,2])-np.sqrt(samples[:,0]/samples[:,2])
        gain[name]=dict(gain=float(np.sqrt(daily.b.sum()/daily.n.sum())-np.sqrt(daily.a.sum()/daily.n.sum())),lower=float(np.quantile(dif,.025)),upper=float(np.quantile(dif,.975)))
    ens, gru=metrics[:2]
    promote=all(v['lower']>0 for v in gain.values()) and all(ens[k] is not None and gru[k] is not None and ens[k]>=gru[k] for k in ['precision','recall'])
    out=RUN/'prospective_test';out.mkdir(exist_ok=False)
    result.to_csv(out/'predictions.csv.gz',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    report=dict(status='complete',recommended=promote,paired_gains=gain,eligible_history_rows=before,paired_rows=len(rows),missing_weather_rows=before-len(rows),
        coverage=float(result.y.between(result.lower,result.upper).mean()),protocol_sha256=digest(RUN/'manifest.json'),aligned_sha256=digest(aligned_path),weather_sha256=digest(weather_path))
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['develop','status','evaluate'])
    ap.add_argument('--aligned',type=Path);ap.add_argument('--weather',type=Path);args=ap.parse_args()
    if args.action=='develop': develop()
    elif args.action=='status':
        m=load_frozen(); print(json.dumps(dict(**m,period_finished=prospective_ready(m)),indent=2))
    else:
        if not args.aligned or not args.weather: ap.error('Provide quality-checked --aligned and --weather files with manifests')
        evaluate(args.aligned,args.weather)

if __name__=='__main__': main()
