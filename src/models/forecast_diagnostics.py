"""Diagnostic summaries; no model selection or retraining on benchmark labels."""
import numpy as np
import pandas as pd
from pathlib import Path

def summarize(frame, threshold=50):
    delta=frame.y-frame.reference
    forecast=frame.pred-frame.reference
    observed=delta.abs()>=threshold
    flagged=forecast.abs()>=threshold
    correct=observed & flagged & (np.sign(delta)==np.sign(forecast))
    def ratio(n,d): return float(n/d) if d else None
    return dict(n=len(frame),RMSE=float(np.sqrt(np.mean((frame.pred-frame.y)**2))),
        baseline_RMSE=float(np.sqrt(np.mean((frame.reference-frame.y)**2))),
        sharp_changes=int(observed.sum()),predicted_sharp_changes=int(flagged.sum()),
        direction_correct_detections=int(correct.sum()),recall=ratio(correct.sum(),observed.sum()),
        precision=ratio(correct.sum(),flagged.sum()),mean_actual_change=float(delta.abs().mean()),
        mean_predicted_change=float(forecast.abs().mean()))

def main():
    root=Path(__file__).resolve().parents[2]
    df=pd.read_csv(root/'results/forecast_2025/runs/locked-2025-v1/predictions.csv.gz')
    df['actual_change']=df.y-df.reference; df['predicted_change']=df.pred-df.reference
    df['absolute_error']=(df.pred-df.y).abs()
    df['change_band']=pd.cut(df.actual_change.abs(),[-1,25,50,np.inf],right=False,labels=['under 25','25 to under 50','50 or more'])
    out=root/'results/forecast_diagnostics'; out.mkdir(exist_ok=True)
    rows=[]
    for kind,col in [('city','city'),('change_band','change_band')]:
        for name,g in df.groupby(col,observed=True): rows.append(dict(group_type=kind,group=str(name),**summarize(g)))
    rows.append(dict(group_type='overall',group='all',**summarize(df)))
    pd.DataFrame(rows).to_csv(out/'spike_summary.csv',index=False)
    df.sort_values('absolute_error',ascending=False).head(250).to_csv(out/'largest_misses.csv',index=False)
    (out/'README.txt').write_text('Frozen 2025 benchmark diagnostics only. Sharp change means absolute observed daily change >=50 AQI. Detection requires predicted change >=50 in the correct direction. Empty precision/recall means no denominator. These labels are now seen; new candidates must be tested on a fresh year.\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
