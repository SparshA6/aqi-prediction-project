"""Finish the current local download/training pipeline, then run the locked test."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pandas as pd
from paths import ROOT
from prepare_2025 import readiness

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outputs',type=Path,required=True)
    ap.add_argument('--timeout-hours',type=float,default=24)
    args=ap.parse_args(); outputs=args.outputs; outputs.mkdir(parents=True,exist_ok=True)
    development=ROOT/'results/forecast_development/runs/2025-ready-v1'
    start=time.monotonic(); state={}; verified=False
    def status(phase,missing=None,error=None):
        state.update(status=phase,updated_utc=datetime.now(timezone.utc).isoformat(),missing_inputs=missing or [],error=error)
        p=outputs/'Forecast_2025_Status.json'; temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(state,indent=2)); temp.replace(p)
        text=['2025 NEXT-DAY FORECAST DEVELOPMENT',f'Updated UTC: {state["updated_utc"]}',f'Status: {phase}',
            'Train: January 2023–September 2024. Select: October 2024. Calibrate: November–December 2024.',
            '2025 is reserved for a single test after downloads and quality checks. The completed 2024 benchmark is preserved.',
            'Inputs: 28-day AQI history; exploratory ERA5-Land weather delayed 7/14 days. Final reanalysis publication vintage is unverified.',
            '17 automated checks passed before pipeline launch. Download completion is not equivalent to model accuracy.']
        if 'selected_model' in state: text.extend([f'Selected model: {state["selected_model"]}',f'October 2024 selection RMSE: {state["selection_rmse"]:.3f}', 'Selection scores are development results, not fresh-test results.'])
        if missing: text.extend([f'Pending input checks: {len(missing)}']+missing)
        if error: text.append(f'Action required: {error}')
        if phase=='complete': text.extend([f'2025 RMSE: {state["test_rmse"]:.3f}',f'Persistence RMSE: {state["reference_rmse"]:.3f}',f'Empirical interval coverage: {state["coverage"]:.1%}'])
        p=outputs/'Forecast_2025_Report.txt'; temp=p.with_suffix('.tmp'); temp.write_text('\n'.join(text)+'\n',encoding='utf-8'); temp.replace(p)
    try:
        while True:
            missing=readiness()
            mpath=development/'manifest.json'
            frozen=json.loads(mpath.read_text()) if mpath.exists() else {}
            ready=frozen.get('status')=='frozen_awaiting_2025'
            if ready:
                # The initial worker was already importing when provenance hashing was
                # added. Seal its saved models before any test data are loaded.
                if 'model_sha256' not in frozen:
                    frozen['model_sha256']={str(p.relative_to(development)):sha256(p.read_bytes()).hexdigest() for p in (development/'models').iterdir()}
                    frozen['model_hash_sealed_utc']=datetime.now(timezone.utc).isoformat()
                    temp=development/'manifest.tmp'; temp.write_text(json.dumps(frozen,indent=2)); temp.replace(mpath)
                metrics=pd.read_csv(development/'development_metrics.csv')
                chosen=metrics[metrics.split=='selection'].sort_values(['RMSE','model']).iloc[0]
                if chosen.model!=frozen['selected_model']: raise ValueError('Saved selection disagrees with validation scores')
                if not verified:
                    subprocess.run([sys.executable,'-u','src/models/verify_forecast_development.py'],cwd=ROOT,check=True)
                    shutil.copy2(development/'verification.json',outputs/'Forecast_Development_Verification.json')
                    verified=True
                state.update(selected_model=frozen['selected_model'],selection_rmse=float(chosen.RMSE))
                shutil.copy2(mpath,outputs/'Forecast_Development_Manifest.json')
                shutil.copy2(development/'development_metrics.csv',outputs/'Forecast_Development_Metrics.csv')
            status('waiting_for_2025_data' if ready else 'training_and_downloading',missing)
            if ready and not missing: break
            if time.monotonic()-start>args.timeout_hours*3600:
                status('needs_resume',missing,'Download/training wait expired; rerun incomplete downloads and this pipeline.'); return
            print(f'{datetime.now(timezone.utc).isoformat()}: training frozen={ready}; missing inputs={len(missing)}',flush=True)
            time.sleep(60)
        status('checking_2025_data')
        subprocess.run([sys.executable,'-u','src/prepare_2025.py','--resume'],cwd=ROOT,check=True)
        status('evaluating_frozen_forecast')
        subprocess.run([sys.executable,'-u','src/models/evaluate_forecast_2025.py','--device','cuda'],cwd=ROOT,check=True)
        result=ROOT/'results/forecast_2025/runs/locked-2025-v1'
        m=json.loads((result/'manifest.json').read_text()); scores=pd.read_csv(result/'metrics.csv')
        state.update(test_rmse=float(scores.iloc[0].RMSE),reference_rmse=float(scores.iloc[1].RMSE),coverage=m['empirical_interval_coverage'])
        for name,dest in [('metrics.csv','Forecast_2025_Metrics.csv'),('group_metrics.csv','Forecast_2025_City_Season_Metrics.csv'),('manifest.json','Forecast_2025_Manifest.json'),('predictions.csv.gz','Forecast_2025_Predictions.csv.gz')]: shutil.copy2(result/name,outputs/dest)
        status('complete')
    except Exception as exc:
        status('failed',error=str(exc)); raise

if __name__=='__main__': main()
