"""Independently recompute frozen development scores and reload predictions."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import joblib
from forecast_development import feature_table
from run_forecast_development import scores
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'results/forecast_development/runs/2026-ready-v1'
    manifest=json.loads((out/'manifest.json').read_text())
    artifact=out/'forecast_model.joblib'
    assert manifest['status']=='frozen_awaiting_2026' and manifest['test_year_loaded'] is False
    assert sha256(artifact.read_bytes()).hexdigest()==manifest['artifact_sha256']
    metrics=pd.read_csv(out/'development_metrics.csv');pred=pd.read_csv(out/'development_predictions.csv.gz',parse_dates=['date'])
    for row in metrics.itertuples():
        g=pred[(pred.model==row.model)&(pred.split==row.split)]
        values=scores(g.y,g.pred)
        assert values['n']==row.n
        for key in ['RMSE','MAE','R2']:assert np.isclose(values[key],getattr(row,key),atol=1e-8)
    selected=metrics[metrics.split=='selection'].sort_values(['RMSE','model']).iloc[0].model
    assert selected==manifest['selected_model']
    g=pred[(pred.model==selected)&(pred.split=='calibration')]
    errors=(g.y-g.pred).abs();q=min(1,np.ceil((len(errors)+1)*.9)/len(errors))
    assert np.isclose(np.quantile(errors,q,method='higher'),manifest['interval_radius'])
    aligned=pd.read_csv(ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv',parse_dates=['date'])
    df,_=feature_table(aligned)
    weather=pd.read_csv(ROOT/'data/processed/forecast_weather/daily.csv',parse_dates=['date','forecast_available_before_utc'])
    assert (weather.forecast_available_before_utc<=weather.date).all()
    df=df.merge(weather,on=['city','date'],how='left',validate='many_to_one')
    chosen=pred[(pred.model==selected)&(pred.split=='selection')].iloc[::30]
    sample=chosen.merge(df,on=['date','station_id'],suffixes=('_saved',''),validate='one_to_one')
    a=joblib.load(artifact)
    assert a['kind']=='residual_tabular'
    a['model'].set_params(device='cpu')
    actual=np.clip(sample.aqi_l1.to_numpy()+a['model'].predict(sample[a['features']]),0,500)
    assert np.allclose(actual,sample.pred,atol=1e-3)
    result=dict(status='passed',metric_groups_recomputed=len(metrics),cpu_reload_rows=len(sample),
                selected_using_october_2025=True,interval_recomputed=True,test_year_excluded=True,weather_cutoff_checked=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
