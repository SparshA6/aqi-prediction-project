"""Independent checks of the completed locked 2025 forecast evaluation."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from forecast_development import feature_table
from evaluate_forecast_2025 import predict,scores
ROOT=Path(__file__).resolve().parents[2]

def main():
    run=ROOT/'results/forecast_2025/runs/locked-2025-v1'
    m=json.loads((run/'manifest.json').read_text())
    if m['status']!='complete': raise ValueError('Evaluation is incomplete')
    development=ROOT/'results/forecast_development/runs'/m['frozen_development_run']
    frozen=json.loads((development/'manifest.json').read_text())
    if datetime.fromisoformat(frozen['frozen_utc'])>=datetime.fromisoformat(m['started_utc']): raise ValueError('Model was not frozen before testing')
    if frozen['selected_model']!=m['selected_model']: raise ValueError('Test selection changed')
    for file,expected in {**frozen['model_sha256'],'forecast_model.joblib':frozen['artifact_sha256']}.items():
        if sha256((development/file).read_bytes()).hexdigest()!=expected: raise ValueError(f'Model changed: {file}')
    r=pd.read_csv(run/'predictions.csv.gz',parse_dates=['date'])
    if r.duplicated(['station_id','date']).any() or set(r.date.dt.year)!={2025}: raise ValueError('Invalid test keys/dates')
    if not np.isfinite(r[['y','pred','reference','lower','upper']].to_numpy()).all(): raise ValueError('Nonfinite test outputs')
    table=pd.read_csv(run/'metrics.csv')
    for model,col in ((m['selected_model'],'pred'),('Persistence reference','reference')):
        row=table[table.model==model].iloc[0]
        for key,value in scores(r.y,r[col]).items(): np.testing.assert_allclose(value,row[key],rtol=1e-8,atol=1e-8)
    groups=pd.read_csv(run/'group_metrics.csv')
    for group in ('city','season'):
        for name,g in r.groupby(group):
            for model,col in ((m['selected_model'],'pred'),('Persistence reference','reference')):
                row=groups[(groups.group_type==group)&(groups.group==name)&(groups.model==model)].iloc[0]
                for key,value in scores(g.y,g[col]).items(): np.testing.assert_allclose(value,row[key],rtol=1e-8,atol=1e-8)
    calibration=pd.read_csv(development/'development_predictions.csv.gz')
    calibration=calibration[(calibration.split=='calibration')&(calibration.model==m['selected_model'])]
    residual=np.abs(calibration.y-calibration.pred); radius=float(np.quantile(residual,min(1.,np.ceil((len(residual)+1)*.9)/len(residual)),method='higher'))
    np.testing.assert_allclose(radius,frozen['interval_radius'])
    np.testing.assert_allclose(r.lower,np.clip(r.pred-radius,0,500)); np.testing.assert_allclose(r.upper,np.clip(r.pred+radius,0,500))
    np.testing.assert_allclose(((r.y>=r.lower)&(r.y<=r.upper)).mean(),m['empirical_interval_coverage'])
    aligned=pd.read_csv(ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv',parse_dates=['date'])
    df,_=feature_table(aligned)
    sample=r.sample(n=80,random_state=1)
    rows=sample[['station_id','date']].merge(df,on=['station_id','date'],validate='one_to_one')
    np.testing.assert_allclose(rows.aqi_l1,sample.reference.to_numpy())
    artifact=joblib.load(development/'forecast_model.joblib')
    reloaded=predict(artifact,development,aligned,rows,'cpu')
    np.testing.assert_allclose(reloaded,sample.pred.to_numpy(),rtol=1e-4,atol=.02)
    protected=pd.read_csv(ROOT/'data/processed/multiyear_2023_2024/aligned_dataset.csv',parse_dates=['date'])
    pd.testing.assert_frame_equal(aligned[aligned.date.dt.year<2025].reset_index(drop=True),protected,check_dtype=False)
    result=dict(status='passed',checks=['Frozen selection, calibration and model hashes unchanged','Unique eligible 2025 station-days and finite outputs','Overall and every city/season score independently recomputed','Intervals match only 2024 calibration residuals','Empirical coverage independently recomputed','80 saved forecasts reproduced from CPU checkpoints','Persistence equals previous calendar day at sampled stations','Protected 2023-2024 rows unchanged'],sample_reload_max_difference=float(np.max(np.abs(reloaded-sample.pred.to_numpy()))))
    (run/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
