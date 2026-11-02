"""
Run the baseline models over the feature-ladder x model grid for one evaluation protocol.

    python src/models/run_baselines.py --protocol chrono            # train Jan-Aug, val Sep-Oct, test Nov-Dec
    python src/models/run_baselines.py --protocol loco              # leave one city out (8 folds)
    python src/models/run_baselines.py --protocol season_cv         # 4 interleaved-month folds
    options: --models Ridge XGBoost   --sets SW "SW+G"   --jobs 4   --quick (smoke test on a data subsample)

Writes results/baselines/metrics_<protocol>.csv (long format) and predictions_<protocol>.csv.gz.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from baselines import MODEL_NAMES, climatology, fit_predict, persistence
from data import FEATURE_SETS, load_dataset, make_splits
from evaluate import evaluate
from runs import add_run_args, complete_run, select_splits, start_run
from paths import ROOT

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["chrono", "loco", "season_cv"])
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    add_run_args(ap)
    a = ap.parse_args()
    if getattr(a, "seeds", 1) < 1:
        ap.error("--seeds must be positive")

    df = load_dataset()
    if a.quick:
        df = df.sample(4000, random_state=0).sort_values(["date", "station_id"]).reset_index(drop=True)
    splits = select_splits(make_splits(df, a.protocol), a.folds)
    OUT = start_run(a, "baselines", df, splits)
    print(f"{len(df):,} rows | protocol {a.protocol} | models {a.models} | sets {a.sets}", flush=True)

    metrics, preds = [], []

    def record(fold, split, fs, model, d, p):
        for r in evaluate(d, p):
            metrics.append(dict(protocol=a.protocol, fold=fold, split=split, feature_set=fs, model=model, **r))
        preds.append(pd.DataFrame({"protocol": a.protocol, "fold": fold, "split": split, "feature_set": fs, "model": model,
                                   "row": d.index.values, "station_id": d.station_id.values, "date": d.date.values,
                                   "y": d.y.values, "pred": np.asarray(p, float)}))

    t0 = time.time()
    for fold, tr, evs in splits:
        dtr = df.loc[tr]
        # reference baselines (no features / no fitting to the inputs)
        for split, idx in evs.items():
            de = df.loc[idx]
            record(fold, split, "-", "Persistence(yesterday)", de, persistence(de))
            record(fold, split, "-", "Climatology(city-month)", de, climatology(dtr, de))
        for fs in a.sets:
            cols = FEATURE_SETS[fs]
            for model in a.models:
                t1 = time.time()
                out = fit_predict(model, dtr[cols], dtr["y"], {k: df.loc[i, cols] for k, i in evs.items()}, n_jobs=a.jobs)
                for split, idx in evs.items():
                    record(fold, split, fs, model, df.loc[idx], out[split])
                msg = " | ".join(f"{split} RMSE {np.sqrt(np.mean((out[split] - df.loc[idx, 'y'].values) ** 2)):.1f}" for split, idx in evs.items())
                print(f"[{time.time() - t0:6.0f}s] {fold:<18} {fs:<9} {model:<13} {msg}   ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(metrics).to_csv(OUT / f"metrics_{a.protocol}.csv", index=False)      # checkpoint after every fold
        pd.concat(preds).to_csv(OUT / f"predictions_{a.protocol}.csv.gz", index=False)

    complete_run(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
