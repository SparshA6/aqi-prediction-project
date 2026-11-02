"""Local historical AQI dashboard. No live-data or live-forecast claim."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
from hashlib import sha256
import re
import sys

import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[2]
app = Flask(__name__)


@lru_cache(maxsize=12)
def csv(path, modified):
    return pd.read_csv(path)


def read(path):
    return csv(str(path), path.stat().st_mtime_ns).copy()


def fresh_forecast():
    directory=ROOT/'results/forecast_2025/runs/locked-2025-v1'
    if not (directory/'verification.json').exists(): return None
    manifest=json.loads((directory/'manifest.json').read_text())
    verification=json.loads((directory/'verification.json').read_text())
    if manifest.get('status')!='complete' or verification.get('status')!='passed': return None
    return directory,manifest


def state():
    modes = {}
    original = ROOT / "results/spatial"
    same_path = original / "predictions_chrono.csv.gz"
    fs, model = "SWT-w7", "S-SA"
    source = "Historical spatial study; checkpoints predate corrected validation"
    studies = sorted((ROOT / "results/studies").glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    study = json.loads(studies[0].read_text()) if studies else None
    if study:
        for job in study["jobs"]:
            if job["family"] == "spatial" and job["protocol"] == "chrono" and job["status"] == "complete":
                directory = Path(job["results_dir"])
                metrics = read(directory / "metrics_chrono.csv")
                val = metrics[(metrics.split == "val") & (metrics.subset == "has_lag1") & (metrics.level == "overall")]
                chosen = val.loc[val.RMSE.idxmin()]
                fs, model = chosen.feature_set, chosen.model
                same_path = directory / "predictions_chrono.csv.gz"
                original = directory
                source = "Corrected spatial study; model selected on validation error"
    if same_path.exists():
        frame = read(same_path)
        frame = frame[(frame.split == "test") & (frame.feature_set == fs) & (frame.model == model)]
        frame = frame[["date", "station_id", "y", "pred"]]
        metadata = pd.read_csv(ROOT / "config/stations.csv")[["station_id", "station_name", "city"]]
        frame = frame.merge(metadata, on="station_id", how="left")
        metrics = read(original / "metrics_chrono.csv")
        headline = metrics[(metrics.split == "test") & (metrics.subset == "has_lag1") & (metrics.level == "overall") & (metrics.feature_set == fs) & (metrics.model == model)].iloc[0]
        modes["same_day"] = dict(label="Same-day estimate", source=source, model=model, feature_set=fs,
            rmse=float(headline.RMSE), scored_rows=int(headline.n), interval_coverage=None,
            description="Satellite and daily weather describe the same UTC day. This is retrospective estimation.",
            rows=json.loads(frame.to_json(orient="records")))
    forecasts = sorted((ROOT / "results/forecast/runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if forecasts:
        path = forecasts[0]
        manifest = json.loads(path.read_text())
        frame = read(path.parent / "predictions.csv")
        metrics = read(path.parent / "metrics.csv")
        headline = metrics[(metrics.split == "test") & (metrics.model == manifest["selected_model"])].iloc[0]
        modes["next_day"] = dict(label="Next-day forecast", model=manifest["selected_model"], source="Historical forecast; selected using September, calibrated using October",
            rmse=float(headline.RMSE), scored_rows=int(headline.n), interval_coverage=manifest["test_interval_coverage"],
            description="Predicts the next UTC day's AQI from previously observed AQI and known calendar/location inputs. No future weather enters the forecast.",
            rows=json.loads(frame.to_json(orient="records")))
    audits = []
    for path in sorted((ROOT / "results/faithfulness").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
        data = json.loads(path.read_text())
        correlation = path.parent / "correlations.csv"
        if correlation.exists():
            c = read(correlation)
            scores = c.groupby(["baseline", "method"]).spearman.mean().reset_index()
            audits.append(dict(kind=data.get("kind"), model=data.get("model"), rows=data.get("rows"), validation=data.get("validation", "Provenance in saved experiment logs"), scores=json.loads(scores.to_json(orient="records"))))
    jobs = [] if study is None else [dict(family=j["family"], protocol=j["protocol"], status=j["status"]) for j in study["jobs"]]
    year_check = None
    for path in sorted((ROOT / 'results/year_holdout/runs').glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:1]:
        result = json.loads(path.read_text())
        year_check = dict(status=result['status'], protocol=result['protocol'], rows=result.get('rows', {}), modes=[])
        if result['status'] == 'complete':
            scores = read(path.parent / 'metrics.csv')
            for mode, details in result['modes'].items():
                test = scores[(scores['mode'] == mode) & (scores['split'] == 'test')]
                chosen = test[test.model == details['selected_model']].iloc[0]
                reference = test[test.model == 'Persistence'].iloc[0]
                year_check['modes'].append(dict(mode=mode, selected_model=details['selected_model'], rmse=float(chosen.RMSE), reference_rmse=float(reference.RMSE), scored_rows=int(chosen.n), coverage=details['empirical_test_coverage'], gain=details['paired_rmse_gain']))
            verification=path.parent/'verification.json'
            if verification.exists() and json.loads(verification.read_text()).get('status')=='passed':
                details=result['modes']['same_day']
                selected=scores[(scores['mode']=='same_day')&(scores.split=='test')&(scores.model==details['selected_model'])].iloc[0]
                frame=read(path.parent/'predictions.csv.gz')
                frame=frame[(frame['mode']=='same_day')&(frame.model==details['selected_model'])].copy()
                metadata=pd.read_csv(ROOT/'config/stations.csv')[['station_id','station_name']]
                frame=frame.merge(metadata,on='station_id',how='left',validate='many_to_one')
                modes['same_day']=dict(label='Same-day estimate',model=details['selected_model'],
                    source='2024 test · model selected in October 2023; intervals calibrated in November–December 2023',
                    rmse=float(selected.RMSE),scored_rows=int(selected.n),interval_coverage=details['empirical_test_coverage'],
                    description='Retrospective same-day estimate using satellite/weather observations and previous AQI. Selected using 2023, tested on 2024; same-day weather is available after that day ends.',
                    rows=json.loads(frame[['date','station_id','station_name','city','y','pred','reference','lower','upper']].to_json(orient='records')))
    # Select environment-only estimator using its original 2023 selection split.
    if modes.get('same_day') and year_check:
        selection=scores[(scores['mode']=='same_day')&(scores.split=='selection')&scores.model.str.endswith('-env')]
        if len(selection):
            chosen=selection.sort_values(['RMSE','model']).iloc[0].model
            env=read(path.parent/'predictions.csv.gz')
            env=env[(env['mode']=='same_day')&(env.model==chosen)].copy()
            env=env.merge(metadata,on='station_id',how='left',validate='many_to_one')
            modes['same_day_env']=dict(label='Same-day · environment only',model=chosen,
                source='2024 test · environment-only model selected using October 2023',
                description='Retrospective estimate from same-day satellite and weather, without previous AQI. Separate from the history-assisted same-day model.',
                rmse=float(selection.loc[selection.model==chosen,'RMSE'].iloc[0]),scored_rows=len(env),interval_coverage=None,
                rows=json.loads(env[['date','station_id','station_name','city','y','pred','reference']].to_json(orient='records')))
            modes['same_day_env']['rmse']=float(np.sqrt(np.mean((env.pred-env.y)**2)))
    forecast_check=None
    fresh=fresh_forecast()
    if fresh:
        directory,details=fresh
        scores=read(directory/'metrics.csv'); selected=scores.iloc[0]; reference=scores.iloc[1]
        forecast_check=dict(model=details['selected_model'],rmse=float(selected.RMSE),reference_rmse=float(reference.RMSE),
            scored_rows=int(selected.n),coverage=details['empirical_interval_coverage'],gain=details['paired_rmse_gain'],
            cities=json.loads(read(directory/'group_metrics.csv').query("group_type=='city'").to_json(orient='records')),
            custom_model_approved=details['paired_rmse_gain']['lower']>0)
        frame=read(directory/'predictions.csv.gz')
        modes['next_day']=dict(label='Next-day forecast',model=details['selected_model'],source='2025 test · selected using October 2024; calibrated using November–December 2024',
            rmse=float(selected.RMSE),scored_rows=int(selected.n),interval_coverage=details['empirical_interval_coverage'],
            history_days=28 if forecast_check['custom_model_approved'] else 1,
            description='Rolling next-day forecast for 2025. The frozen model uses AQI received on earlier days; model weights are never fitted on 2025 targets.',
            rows=json.loads(frame[['date','station_id','station_name','city','y','pred','reference','lower','upper']].to_json(orient='records')))
    development_path=ROOT/'results/forecast_development/runs/2026-ready-v1'
    development=None
    if (development_path/'manifest.json').exists():
        development=json.loads((development_path/'manifest.json').read_text())
        if (development_path/'development_metrics.csv').exists():
            development['metrics']=json.loads(read(development_path/'development_metrics.csv').to_json(orient='records'))
        development['downloaded_files']=len(list((ROOT/'data/raw/cpcb_2026').rglob('*2026*.xlsx')))
        development['expected_files']=9*len(read(ROOT/'config/stations.csv'))
        result=ROOT/'results/forecast_2026/runs/locked-2026-v1'
        development['test_status']='pending'
        if (result/'verification.json').exists() and json.loads((result/'verification.json').read_text()).get('status')=='passed':
            development['test_status']='verified'
            metrics=read(result/'metrics.csv')
            development['test_metrics']=json.loads(metrics.to_json(orient='records'))
            report=json.loads((result/'manifest.json').read_text())
            development.update(recommended=report.get('recommended',False),coverage_stations=report.get('stations'),
                excluded_stations=report.get('excluded_stations',[]),gain_vs_gru=report.get('paired_gru_rmse_gain'),
                gain_vs_persistence=report.get('paired_rmse_gain'),interval_coverage=report['interval_coverage'],
                city_metrics=json.loads(read(result/'city_metrics.csv').to_json(orient='records')),
                spike_metrics=json.loads(read(result/'spike_metrics.csv').to_json(orient='records')) if (result/'spike_metrics.csv').exists() else [])
            frame=read(result/'predictions.csv.gz')
            available=read(ROOT/'data/processed/forecast_2026/coverage.csv')
            missing_ids=set(available.station_id)-set(frame.station_id)
            unscored=read(ROOT/'config/stations.csv')
            unscored=unscored[unscored.station_id.isin(missing_ids)][['station_id','station_name','city']]
            development['unscored_stations']=json.loads(unscored.to_json(orient='records'))
            modes['next_day_2026']=dict(label='Next-day · weather model · 2026',model=report['selected_model'],
                source='Jan–Sep 2026 test · selected using October 2025; calibrated November–December 2025',
                description=f"Frozen weather-assisted forecast. Uses earlier AQI and archived 24-hour-ahead GFS forecasts. Tested on {report.get('stations','available')} of 107 stations; coverage gaps are listed below. No fitting on 2026 targets. Archived ingestion latency is unverified.",
                rmse=float(metrics.iloc[0].RMSE),scored_rows=len(frame),interval_coverage=report['interval_coverage'],
                rows=json.loads(frame.to_json(orient='records')))
    research_path=ROOT/'results/followthrough/oct12-v1/summary.json'
    research=json.loads(research_path.read_text()) if research_path.exists() else None
    return dict(prospective=prospective_state(),research=research,default_mode='next_day_2026' if development and development.get('recommended') else 'next_day',development_2026=development,stations=json.loads(read(ROOT/'config/stations.csv')[['station_id','station_name','city']].to_json(orient='records')),modes=modes, jobs=jobs, study_status=study["status"] if study else "not_started", audits=audits,
                forecast_2025=forecast_check,
                year_holdout=year_check,
                training_device=study.get("training_device", "cpu") if study else "cpu", seeds=study.get("seeds", 3) if study else 3,
                disclaimer="Historical results: same-day uses 2024; GRU next-day uses 2025; the separate weather-model test uses Jan–Sep 2026 when verified.")


@app.get("/")
def home():
    return send_file(Path(__file__).with_name("index.html"))


@app.get("/api/state")
def data():
    return jsonify(state())


@app.post("/api/forecast")
def forecast():
    """Experimental inference from user-supplied observations; no future retrievals."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="Provide a forecast input object"), 400
    try:
        latest = float(payload["latest_aqi"])
        if not np.isfinite(latest) or not 0 <= latest <= 500:
            raise ValueError("AQI must be between 0 and 500")
        observed_date = pd.Timestamp(payload["observed_date"])
        if pd.isna(observed_date):
            raise ValueError("Choose the date of the last observed UTC day")
        observed_date = observed_date.normalize()
        stations = pd.read_csv(ROOT / "config/stations.csv").set_index("station_id")
        station_id = payload.get("station_id")
        if station_id not in stations.index:
            raise ValueError("Choose a known station")
        fresh=fresh_forecast()
        if fresh and fresh[1]['paired_rmse_gain']['lower']>0:
            directory,manifest=fresh
            development=ROOT/'results/forecast_development/runs'/manifest['frozen_development_run']
            frozen=json.loads((development/'manifest.json').read_text())
            artifact_path=development/'forecast_model.joblib'
            if sha256(artifact_path.read_bytes()).hexdigest()!=frozen['artifact_sha256']: raise ValueError('Saved forecast model changed')
            artifact=joblib.load(artifact_path)
            if artifact['kind']!='gru_ensemble': raise ValueError('This forecast input form requires a history model')
            values=payload.get('previous_aqi')
            if not isinstance(values,list) or len(values)!=27: raise ValueError('Enter 27 previous daily AQI values, oldest first, plus the latest value')
            previous=np.asarray([float(v) for v in values],dtype=float)
            if not np.isfinite(previous).all() or ((previous<0)|(previous>500)).any(): raise ValueError('Each previous AQI must be between 0 and 500')
            target_date=observed_date+pd.Timedelta(days=1)
            history_dates=pd.date_range(end=observed_date,periods=28)
            aligned=pd.DataFrame(dict(date=history_dates,station_id=station_id,AQI_eod=[*previous,latest],aqi_flat_day=False))
            rows=pd.DataFrame(dict(date=[target_date],station_id=[station_id],lat=[stations.loc[station_id,'lat']],lon=[stations.loc[station_id,'lon']]))
            sys.path.insert(0,str(ROOT/'src/models'))
            from evaluate_forecast_2025 import predict
            for checkpoint in artifact['checkpoints']:
                file=str(Path('models')/checkpoint['weight'])
                if sha256((development/file).read_bytes()).hexdigest()!=frozen['model_sha256'][file]: raise ValueError('Saved model checkpoint changed')
            prediction=float(predict(artifact,development,aligned,rows,'cpu')[0])
            radius=frozen['interval_radius']
            return jsonify(dict(target_date=target_date.date().isoformat(),prediction=prediction,lower=max(0.,prediction-radius),upper=min(500.,prediction+radius),
                model=manifest['selected_model'],empirical_backtest_coverage=manifest['empirical_interval_coverage'],
                note='Forecast from your supplied 28 days of AQI. Model tested on 2025; this is not a verified live reading.'))
        manifests = sorted((ROOT / "results/forecast/runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not manifests:
            raise ValueError("No trained forecast is available")
        artifact = joblib.load(manifests[0].parent / "forecast_model.joblib")
        target_date = observed_date + pd.Timedelta(days=1)
        history = []
        for key in ("lag1", "lag2", "lag7"):
            value = float(payload[key]) if payload.get(key) not in (None, "") else np.nan
            if np.isfinite(value) and not 0 <= value <= 500:
                raise ValueError("Lagged AQI must be between 0 and 500")
            history.append(value)
        row = dict(latest_aqi=latest, lag1=history[0], lag2=history[1], lag7=history[2],
                   mean3=float(np.nanmean([latest, *history[:2]])), lat=stations.loc[station_id, "lat"], lon=stations.loc[station_id, "lon"],
                   doy_sin=np.sin(2*np.pi*target_date.dayofyear/365.25), doy_cos=np.cos(2*np.pi*target_date.dayofyear/365.25))
        prediction = latest if artifact["model"] is None else float(artifact["model"].predict(pd.DataFrame([row])[artifact["features"]])[0])
        prediction = float(np.clip(prediction, 0, 500))
        manifest = json.loads(manifests[0].read_text())
        return jsonify(dict(target_date=target_date.date().isoformat(), prediction=prediction,
                            lower=max(0., prediction-artifact["interval_radius"]), upper=min(500.,prediction+artifact["interval_radius"]),
                            model=artifact["selected"], empirical_backtest_coverage=manifest["test_interval_coverage"],
                            note="Experimental forecast from your supplied observation. Model was evaluated on 2023 data; this is not a verified live reading."))
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


@app.get('/api/download/2025/<kind>')
def download_2025(kind):
    files={'metrics':'metrics.csv','cities':'group_metrics.csv','predictions':'predictions.csv.gz','report':'manifest.json'}
    fresh=fresh_forecast()
    if kind not in files or not fresh: return jsonify(error='Verified 2025 results are not available'),404
    return send_file(fresh[0]/files[kind],as_attachment=True)


@app.get('/api/download/2026/<kind>')
def download_2026(kind):
    files={'metrics':'metrics.csv','cities':'city_metrics.csv','spikes':'spike_metrics.csv','predictions':'predictions.csv.gz','report':'manifest.json'}
    directory=ROOT/'results/forecast_2026/runs/locked-2026-v1'
    check=directory/'verification.json'
    if kind not in files or not check.exists() or json.loads(check.read_text()).get('status')!='passed':
        return jsonify(error='Verified 2026 results are not available'),404
    return send_file(directory/files[kind],as_attachment=True)


@app.post('/api/city-forecast')
def city_forecast():
    """Batch inference from saved observations strictly before target UTC day."""
    payload=request.get_json(silent=True) or {}
    try:
        if not isinstance(payload,dict): raise ValueError('Provide city and observed_date')
        stations=read(ROOT/'config/stations.csv')
        city=payload['city']
        stations=stations[stations.city==city]
        if stations.empty: raise ValueError('Choose a known city')
        end=pd.Timestamp(payload['observed_date'])
        if pd.isna(end) or end.tzinfo is not None: raise ValueError('Choose an unzoned UTC calendar date')
        end=end.normalize(); target=end+pd.Timedelta(days=1)
        aligned=read(ROOT/'data/processed/multiyear_2023_2024_2025/aligned_dataset.csv')
        aligned['date']=pd.to_datetime(aligned.date)
        # Restrict first: no target-day observations enter this inference.
        aligned=aligned[(aligned.date<=end)&(aligned.date>end-pd.Timedelta(days=28))&aligned.station_id.isin(stations.station_id)]
        aligned=aligned[aligned.AQI_eod.notna()&~aligned.aqi_flat_day]
        counts=aligned.groupby('station_id').date.nunique()
        ids=counts[counts==28].index
        stations=stations[stations.station_id.isin(ids)].copy()
        if stations.empty: raise ValueError('No station has 28 valid consecutive saved days ending on that date')
        fresh=fresh_forecast()
        if not fresh: raise ValueError('Verified forecast is unavailable')
        manifest=fresh[1]; dev=ROOT/'results/forecast_development/runs'/manifest['frozen_development_run']
        frozen=json.loads((dev/'manifest.json').read_text())
        artifact_path=dev/'forecast_model.joblib'
        if sha256(artifact_path.read_bytes()).hexdigest()!=frozen['artifact_sha256']: raise ValueError('Model integrity check failed')
        artifact=joblib.load(artifact_path)
        for checkpoint in artifact['checkpoints']:
            file=str(Path('models')/checkpoint['weight'])
            if sha256((dev/file).read_bytes()).hexdigest()!=frozen['model_sha256'][file]: raise ValueError('Checkpoint integrity check failed')
        rows=stations.rename(columns={'lat':'lat','lon':'lon'})[['station_id','station_name','lat','lon']].copy()
        rows['date']=target
        sys.path.insert(0,str(ROOT/'src/models'))
        from evaluate_forecast_2025 import predict
        rows['prediction']=predict(artifact,dev,aligned,rows,'cpu')
        rows['lower']=np.maximum(0,rows.prediction-frozen['interval_radius'])
        rows['upper']=np.minimum(500,rows.prediction+frozen['interval_radius'])
        expected=int((read(ROOT/'config/stations.csv').city==city).sum())
        return jsonify(city=city,target_date=target.date().isoformat(),observed_date=end.date().isoformat(),
            prediction=float(rows.prediction.mean()),available_stations=len(rows),expected_stations=expected,
            model=manifest['selected_model'],stations=json.loads(rows[['station_id','station_name','prediction','lower','upper']].to_json(orient='records')),
            note='Equal-weight average of station forecasts from saved historical observations. No calibrated city interval. Not live AQI.')
    except (ValueError,KeyError,TypeError) as exc:
        return jsonify(error=str(exc)),400


@app.get('/api/download/research/<kind>')
def download_research(kind):
    files={'report':'Results_Report.txt','errors':'error_groups.csv','misses':'largest_errors.csv',
           'attention':'attention_city.csv','deletion':'attention_deletion_city.csv','summary':'summary.json'}
    directory=ROOT/'results/followthrough/oct12-v1'
    if kind not in files or not (directory/files[kind]).exists():
        return jsonify(error='Research output is not available'),404
    return send_file(directory/files[kind],as_attachment=True)


def prospective_state():
    path=ROOT/'results/prospective/status.json'
    info=json.loads(path.read_text()) if path.exists() else dict(status='waiting_inputs',records=[],cities=[],recorded_days=0)
    stations=read(ROOT/'config/stations.csv')
    downloads=[]
    for r in stations.itertuples():
        relative=Path(r.data_dir).relative_to(Path('data/raw/cpcb'))
        files=list((ROOT/'data/raw/cpcb_prospective'/relative).glob('*.xlsx'))
        downloads.append(dict(city=r.city,station_id=r.station_id,files=len(files)))
    live=pd.DataFrame(downloads)
    info['downloaded_stations']=int((live.files>0).sum())
    info['expected_stations']=len(stations)
    next_day=pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()+pd.Timedelta(days=1)
    info['next_target_date']=str(next_day.date()) if pd.Timestamp('2026-10-13')<=next_day<=pd.Timestamp('2026-12-31') else None
    info['download_city_counts']={str(city):int((g.files>0).sum()) for city,g in live.groupby('city')}
    logs=[p for p in [ROOT/'logs/prospective_oct13.log',*(ROOT/'logs').glob('cpcb_prospective*.log')] if p.exists()]
    if logs:
        text=max(logs,key=lambda p:p.stat().st_mtime_ns).read_text(encoding='utf-8',errors='replace')
        info['download_completed']='                    COMPLETE' in text
        info['attempted_stations']=len(re.findall(r'\d+/\d+ months downloaded',text))
        info['october_rows_unavailable']=text.count('Could not find row for October')
        info['downloader_waiting_captcha']='Auto mode: solve' in text and 'CAPTCHA solved.' not in text and 'No CAPTCHA appeared' not in text
    return info


@app.get('/api/prospective')
def prospective_status():
    return jsonify(prospective_state())


@app.get('/api/download/prospective/<date>')
def prospective_predictions(date):
    try:
        target=pd.Timestamp(date)
        if target.tzinfo is not None or str(target.date())!=date:raise ValueError('Invalid date')
        directory=ROOT/'results/prospective/ledger'/date
        m=json.loads((directory/'manifest.json').read_text())
        if m.get('status')!='recorded_before_target' or sha256((directory/'predictions.csv').read_bytes()).hexdigest()!=m['predictions_sha256']:
            raise ValueError('Recorded forecast integrity check failed')
        return send_file(directory/'predictions.csv',as_attachment=True)
    except (ValueError,FileNotFoundError,KeyError):
        return jsonify(error='No verified before-target forecast record is available for that date'),404


@app.post('/api/prospective/record')
def capture_prospective_forecast():
    payload=request.get_json(silent=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('target_date'),str):
        return jsonify(error='Choose a UTC target date'),400
    sys.path.insert(0,str(ROOT/'src'))
    from prospective_tracker import target_day,now_utc,collect_weather,collect_ground,record,refresh_status
    try:
        target=target_day(payload['target_date'])
        if now_utc()>=target.tz_localize('UTC'):raise ValueError('Target day has already started; late forecasts cannot be logged as prospective')
        if (ROOT/'results/prospective/ledger'/str(target.date())).exists():raise ValueError('A record already exists for this target')
        collect_weather(str(target.date()));collect_ground()
        directory=record(str(target.date()));refresh_status()
        return jsonify(status='recorded_before_target',target_date=str(target.date()),download='/api/download/prospective/'+str(target.date()))
    except (ValueError,FileNotFoundError) as e:
        refresh_status();return jsonify(error=str(e)),400


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--export", type=Path)
    args = ap.parse_args()
    if args.export:
        html = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
        payload = json.dumps(state()).replace("<", "\\u003c")
        html = html.replace("/* SNAPSHOT */", "window.snapshot=" + payload + ";")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(html, encoding="utf-8")
        print(f"Dashboard exported: {args.export}")
    else:
        app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
