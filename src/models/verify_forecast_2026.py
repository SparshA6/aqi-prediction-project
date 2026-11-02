"""Independent publication check for actual targets, baseline, metrics and coverage."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'results/forecast_2026/runs/locked-2026-v1'
    manifest=json.loads((out/'manifest.json').read_text())
    df=pd.read_csv(out/'predictions.csv.gz',parse_dates=['date'])
    assert manifest['status']=='complete' and df.date.dt.year.eq(2026).all()
    assert not df.duplicated(['station_id','date']).any()
    fresh=pd.read_csv(ROOT/'data/processed/forecast_2026/aligned_dataset.csv',parse_dates=['date'])
    history=pd.read_csv(ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv',parse_dates=['date'])
    a=pd.concat([history,fresh],ignore_index=True)
    a=a[a.AQI_eod.notna()&~a.aqi_flat_day]
    lookup=a.set_index(['station_id','date']).AQI_eod
    targets=lookup.reindex(pd.MultiIndex.from_arrays([df.station_id,df.date])).to_numpy()
    baseline=lookup.reindex(pd.MultiIndex.from_arrays([df.station_id,df.date-pd.Timedelta(days=1)])).to_numpy()
    assert np.allclose(targets,df.y,atol=1e-9) and np.allclose(baseline,df.reference,atol=1e-9)
    metrics=pd.read_csv(out/'metrics.csv');cities=pd.read_csv(out/'city_metrics.csv')
    count=0
    for name,key in [(manifest['selected_model'],'pred'),('Persistence reference','reference'),('Frozen GRU-28days','gru_pred')]:
        for city,g in [('all',df),*list(df.groupby('city'))]:
            r=metrics[metrics.model==name].iloc[0] if city=='all' else cities[(cities.city==city)&(cities.model==name)].iloc[0]
            e=g[key].to_numpy()-g.y.to_numpy()
            assert len(g)==r.n
            assert np.isclose(np.sqrt(np.mean(e**2)),r.RMSE,atol=1e-9)
            assert np.isclose(np.mean(np.abs(e)),r.MAE,atol=1e-9)
            count+=1
    development=ROOT/'results/forecast_development/runs'/manifest['development_run']
    frozen=json.loads((development/'manifest.json').read_text())
    assert sha256((development/'forecast_model.joblib').read_bytes()).hexdigest()==manifest['development_artifact_sha256']==frozen['artifact_sha256']
    assert not frozen['test_year_loaded']
    assert np.allclose(df.lower,np.maximum(0,df.pred-frozen['interval_radius']))
    assert np.allclose(df.upper,np.minimum(500,df.pred+frozen['interval_radius']))
    assert np.isclose(((df.y>=df.lower)&(df.y<=df.upper)).mean(),manifest['interval_coverage'])
    excluded={r['station_id'] for r in manifest['excluded_stations']}
    assert not excluded.intersection(set(df.station_id))
    result=json.loads((out/'verification.json').read_text())
    result.update(status='passed',independent_publication_check=True,metric_groups_recomputed=count,actual_targets_verified=True,
        exact_calendar_baseline_verified=True,intervals_verified=True,coverage_exclusions_verified=True,rows=len(df))
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
