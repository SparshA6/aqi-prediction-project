"""Post-test diagnostics and city-wise attention reporting; never select models."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/models'))
from forecast_diagnostics import summarize

OUT=ROOT/'results/followthrough/oct12-v1'
SEASONS={12:'winter',1:'winter',2:'winter',3:'pre-monsoon',4:'pre-monsoon',5:'pre-monsoon',6:'monsoon',7:'monsoon',8:'monsoon',9:'monsoon',10:'post-monsoon',11:'post-monsoon'}

def diagnostics(frame):
    r=frame.copy();r['date']=pd.to_datetime(r.date)
    r['season']=r.date.dt.month.map(SEASONS)
    r['aqi_band']=pd.cut(r.y,[-1,50,100,200,300,400,500],labels=['0-50','51-100','101-200','201-300','301-400','401-500'])
    r['change_band']=pd.cut(abs(r.y-r.reference),[-1,25,50,np.inf],right=False,labels=['under 25','25-under 50','50 or more'])
    metrics=[];misses=[]
    for model,col in [('Weather XGBoost','pred'),('Frozen GRU','gru_pred'),('Persistence','reference')]:
        g=r.assign(pred=r[col],model=model)
        g['absolute_error']=abs(g.pred-g.y);g['signed_error']=g.pred-g.y
        for group in ['city','season','aqi_band','change_band']:
            for name,part in g.groupby(group,observed=True):
                metrics.append(dict(model=model,group_type=group,group=str(name),bias=float(part.signed_error.mean()),MAE=float(part.absolute_error.mean()),**summarize(part)))
        metrics.append(dict(model=model,group_type='overall',group='all',bias=float(g.signed_error.mean()),MAE=float(g.absolute_error.mean()),**summarize(g)))
        misses.append(g.sort_values(['absolute_error','date','station_id'],ascending=[False,True,True]).head(100))
    return pd.DataFrame(metrics),pd.concat(misses,ignore_index=True)

def attention_reports():
    frames=[];deletions=[];sources=[]
    # Use corrected/purged group audits only; do not blend in legacy checkpoints.
    for kind in ['feature','spatial']:
        available=[]
        for path in (ROOT/'results/faithfulness').glob(kind+'-*/manifest.json'):
            m=json.loads(path.read_text())
            if m.get('validation')=='purged date validation':available.append((path,m))
        if not available:raise ValueError('Missing corrected '+kind+' audit')
        path,m=max(available,key=lambda pair:pair[0].stat().st_mtime_ns)
        for ck,expected in m['checkpoint_sha256'].items():
            if sha256(Path(ck).read_bytes()).hexdigest()!=expected:raise ValueError('Audit checkpoint changed')
        c=pd.read_csv(path.parent/'correlations.csv')
        # Average seeds within each sampled input first, avoiding threefold sample counts.
        per=c.groupby(['row','city','baseline','method'],as_index=False).spearman.mean()
        grouped=per.groupby(['city','baseline','method']).spearman.agg(['count','mean','median','std']).reset_index()
        grouped=grouped.rename(columns={'count':'valid_rows','mean':'mean_spearman','median':'median_spearman','std':'row_std'})
        grouped['kind']=kind;grouped['model']=m['model'];grouped['sampled_rows']=m['rows'];frames.append(grouped)
        d=pd.read_csv(path.parent/'deletion_rows.csv.gz')
        cities=c[['row','city']].drop_duplicates()
        if cities.duplicated('row').any():raise ValueError('Ambiguous audit row city')
        d=d.merge(cities,on='row',validate='many_to_one')
        # Per-row mean over seeds, then city summary.
        d=d.groupby(['row','city','baseline','method','groups_replaced'],as_index=False).change.mean()
        d=d.groupby(['city','baseline','method','groups_replaced']).change.agg(['count','mean']).reset_index()
        d=d.rename(columns={'count':'sampled_rows','mean':'mean_prediction_change'});d['kind']=kind;d['model']=m['model'];deletions.append(d)
        sources.append(dict(kind=kind,model=m['model'],manifest=str(path.relative_to(ROOT)),sha256=sha256(path.read_bytes()).hexdigest(),rows=m['rows'],validation=m['validation']))
    return pd.concat(frames,ignore_index=True),pd.concat(deletions,ignore_index=True),sources

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    path=ROOT/'results/forecast_2026/runs/locked-2026-v1/predictions.csv.gz'
    check=json.loads(path.with_name('verification.json').read_text())
    if check.get('status')!='passed':raise ValueError('Benchmark verification is required')
    frame=pd.read_csv(path);metrics,misses=diagnostics(frame)
    metrics.to_csv(OUT/'error_groups.csv',index=False);misses.to_csv(OUT/'largest_errors.csv',index=False)
    attention,deletion,sources=attention_reports()
    attention.to_csv(OUT/'attention_city.csv',index=False);deletion.to_csv(OUT/'attention_deletion_city.csv',index=False)
    ensemble=json.loads((ROOT/'results/ensemble/runs/prospective-oct13-v1/manifest.json').read_text())
    summary=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        source_prediction_sha256=sha256(path.read_bytes()).hexdigest(),
        benchmark_period='January-September 2026 (previously examined)',
        error_groups=json.loads(metrics.to_json(orient='records')),
        attention_city=json.loads(attention.to_json(orient='records')),attention_sources=sources,
        prospective_ensemble={k:ensemble[k] for k in ['status','weather_weight','gru_weight','interval_radius','prospective_period','tested_on_fresh_data','selection_rule','promotion_rule']})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['AQI PROJECT: ERROR ANALYSIS, ENSEMBLE DEVELOPMENT AND ATTENTION RESULTS','12 October 2026','',
        '1. PAIRED HISTORICAL FORECAST DIAGNOSTICS',
        'All models below use exactly the same 20,086 January-September 2026 station-days. These targets are already seen: diagnostic results are not fresh evidence for a new ensemble.',
        metrics[metrics.group_type=='overall'][['model','n','RMSE','MAE','bias','recall','precision']].to_string(index=False),
        '', 'Error by city:',metrics[metrics.group_type=='city'][['group','model','n','RMSE','MAE','bias','recall','precision']].to_string(index=False),
        '', 'Error by season:',metrics[metrics.group_type=='season'][['group','model','n','RMSE','bias']].to_string(index=False),
        '', 'Error by observed AQI band:',metrics[metrics.group_type=='aqi_band'][['group','model','n','RMSE','bias']].to_string(index=False),
        '', 'Error by actual daily change:',metrics[metrics.group_type=='change_band'][['group','model','n','RMSE','bias']].to_string(index=False),
        '', '2. GRU-WEATHER ENSEMBLE',
        f"Frozen blend: {ensemble['gru_weight']:.0%} GRU + {ensemble['weather_weight']:.0%} weather XGBoost. Weights use October 2025 only; interval calibration uses November-December 2025.",
        'Twenty-one weights were considered (weather share 0 to 1, steps 0.05). Only candidates with no sharp-change precision or recall loss against GRU on selection data were eligible; lowest RMSE wins, and lower weather weight breaks ties.',
        'Selection scores are developmental, not independent generalization evidence. The frozen GRU remains the default. No new ensemble test score is calculated from the already-seen January-September 2026 targets.',
        '', '3. PROSPECTIVE TEST',
        'Target dates: 13 October-31 December 2026. Final scoring opens on 1 January 2027 UTC and requires quality-checked targets plus matching archived earlier-issued weather forecasts. The last endpoint may need January ground exports.',
        'All comparators must use identical station-days. Missing weather excludes a row from every comparator and is counted. Promotion requires positive lower 95% paired calendar-week RMSE gain bounds against both GRU and persistence AND no loss of sharp-change recall or precision versus GRU.',
        'Run: python src/models/ensemble_followthrough.py status',
        'When data are ready: python src/models/ensemble_followthrough.py evaluate --aligned PATH_TO_ALIGNED_CSV --weather PATH_TO_DAILY_WEATHER_CSV',
        'Each input directory must include manifest.json with status=quality_checked and dataset_sha256 matching its CSV. AQI input must include history, station metadata and existing aligned feature columns. The evaluator refuses early evaluation or overwriting its test directory.',
        '', '4. CITY-WISE ATTENTION FAITHFULNESS',
        'These are corrected purged-validation FA and S-SA checkpoint audits, each sampled over 300 input rows. Seed-specific correlations are averaged within each input before city means; valid_rows counts finite row-level correlations, not seeds.',
        attention[['kind','model','city','baseline','method','valid_rows','mean_spearman','median_spearman']].to_string(index=False),
        'Attention and SHAP rankings are each compared against replacement-based sensitivity. Higher Spearman values indicate greater ranking agreement for the chosen replacement scheme. City differences are descriptive and not controlled evidence of different pollution mechanisms.',
        'See attention_deletion_city.csv for city deletion curves. Larger mean output changes after replacing top-ranked groups indicate stronger fixed-model sensitivity; they do not measure accuracy after retraining.',
        'Limitations: sampled observations; correlated inputs; possible out-of-distribution replacement; unequal city sample sizes; seed averaging does not eliminate training uncertainty; spatial sampling restricted to full patches. No causal claims or attribution of these attention maps to Ridge, plain GRU or XGBoost.',
        '', '5. DELIVERABLES',
        'error_groups.csv: overall/city/season/AQI-band/change-band metrics.',
        'largest_errors.csv: top 100 misses per model with target dates and station identifiers.',
        'attention_city.csv and attention_deletion_city.csv: city-wise explanation comparisons.',
        'summary.json: dashboard summary and source provenance.',
        'No new model is promoted before prospective evaluation.'
    ]
    (OUT/'Results_Report.txt').write_text('\n\n'.join(lines),encoding='utf-8')
    print(metrics[metrics.group_type=='overall'].to_string(index=False))
    print('Report:',OUT/'Results_Report.txt')

if __name__=='__main__':main()
