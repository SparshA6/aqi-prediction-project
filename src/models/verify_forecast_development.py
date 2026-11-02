"""Independent saved-score and reload checks using development years only."""
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
    run=ROOT/'results/forecast_development/runs/2025-ready-v1'
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='frozen_awaiting_2025' or manifest['test_year_loaded']: raise ValueError('Run not frozen')
    metrics=pd.read_csv(run/'development_metrics.csv'); predictions=pd.read_csv(run/'development_predictions.csv.gz',parse_dates=['date'])
    if predictions.duplicated(['station_id','date','model','split']).any() or (predictions.date.dt.year>=2025).any(): raise ValueError('Invalid development prediction keys/years')
    for (model,split),g in predictions.groupby(['model','split']):
        row=metrics[(metrics.model==model)&(metrics.split==split)].iloc[0]
        for key,value in scores(g.y,g.pred).items():
            if not np.isclose(value,row[key],rtol=1e-8,atol=1e-8): raise ValueError(f'Score mismatch {model}/{split}/{key}')
    aligned=pd.read_csv(ROOT/'data/processed/multiyear_2023_2024/aligned_dataset.csv',parse_dates=['date'])
    df,_=feature_table(aligned)
    chosen=predictions[predictions.model==manifest['selected_model']].sample(n=80,random_state=0)
    rows=chosen[['station_id','date']].merge(df,on=['station_id','date'],validate='one_to_one')
    artifact=joblib.load(run/'forecast_model.joblib')
    reloaded=predict(artifact,run,aligned,rows,'cpu')
    np.testing.assert_allclose(reloaded,chosen.pred.to_numpy(),rtol=1e-4,atol=.02)
    chosen_model=metrics[metrics.split=='selection'].sort_values(['RMSE','model']).iloc[0].model
    if chosen_model!=manifest['selected_model']: raise ValueError('Selection mismatch')
    original=ROOT/'data/processed/aligned_dataset.csv'
    if sha256(original.read_bytes()).hexdigest()!='26df0c2713cd51573d0a18858638b7247d535f156bba3f3adfbd41db448f9b59': raise ValueError('Protected original changed')
    result=dict(status='passed',checks=['All saved development metrics independently recomputed','Unique station/date/model/split keys','No 2025 development rows','Validation selection matches saved winner','80 selected forecasts reproduced from saved artifacts on CPU','Original 2023 checksum unchanged'],max_reload_difference=float(np.max(np.abs(reloaded-chosen.pred.to_numpy()))))
    (run/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
