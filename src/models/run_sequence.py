"""
Environment-only GRU / LSTM (and the attention variants Uniform / TA / FA / TA+FA, see seq_models.AttnNet) over a sliding window of daily satellite + weather features (no AQI history).

    python src/models/run_sequence.py --protocol chrono
    options: --archs GRU LSTM  --sets SW-w1 SW-w7 SWT-w7  --seeds 3  --threads 2  --quick

Input sets (see seq_data.py):  SW-w1  = today's satellite+weather only (no temporal window: a per-day control)
                               SW-w7  = 7-day window (today + 6 previous days)
                               SWT-w7 = SW-w7 + day-of-year sin/cos at every step
Same splits / metrics / reference baselines as run_baselines.py. Writes results/sequence/*.
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
import torch

from data import load_dataset, make_splits
from evaluate import evaluate
from runs import add_run_args, complete_run, select_splits, start_run
from paths import ROOT
from seq_data import Preprocessor, build_windows
from seq_models import fit_predict_seq
from validation import temporal_validation_split

warnings.filterwarnings("ignore")

SETS = {"SW-w1": (1, False), "SW-w7": (7, False), "SWT-w7": (7, True)}     # name -> (window, use_season)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["chrono", "loco", "season_cv"])
    ap.add_argument("--archs", nargs="+", default=["GRU", "LSTM"], choices=["GRU", "LSTM", "Uniform", "TA", "FA", "TA+FA"])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="sequence", help="results/<out>/ (use 'attention' for the attention variants)")
    ap.add_argument("--save-models", action="store_true", help="store trained weights + preprocessing (for the XAI stage)")
    add_run_args(ap)
    a = ap.parse_args()
    if getattr(a, "seeds", 1) < 1:
        ap.error("--seeds must be positive")
    torch.set_num_threads(a.threads)
    a.device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; install CUDA-enabled PyTorch or choose --device cpu")
    print(f"Training device: {a.device}" + (f" · {torch.cuda.get_device_name(0)}" if a.device == "cuda" else ""), flush=True)

    df = load_dataset()
    if a.quick:
        df = df.sample(4000, random_state=0).sort_values(["date", "station_id"]).reset_index(drop=True)
    X = build_windows(df)
    dates = df.date.values
    y = df.y.values
    splits = select_splits(make_splits(df, a.protocol), a.folds)
    OUT = start_run(a, "sequence", df, splits)
    print(f"{len(df):,} rows | protocol {a.protocol} | archs {a.archs} | sets {a.sets} | seeds {a.seeds}", flush=True)

    metrics, preds = [], []

    def record(fold, split, fs, model, d, p):
        for r in evaluate(d, p):
            metrics.append(dict(protocol=a.protocol, fold=fold, split=split, feature_set=fs, model=model, **r))
        preds.append(pd.DataFrame({"protocol": a.protocol, "fold": fold, "split": split, "feature_set": fs, "model": model,
                                   "row": d.index.values, "station_id": d.station_id.values, "date": d.date.values,
                                   "y": d.y.values, "pred": np.asarray(p, float)}))

    t0 = time.time()
    for fold, tr, evs in splits:
        for fs in a.sets:
            window, use_season = SETS[fs]
            fit_mask, val_mask = temporal_validation_split(dates[tr], window)
            pre = Preprocessor(use_season).fit(X[tr[fit_mask], -window:, :])
            Xtr = pre.transform(X[tr], window)
            Xev = {k: pre.transform(X[i], window) for k, i in evs.items()}
            for arch in a.archs:
                t1 = time.time()
                acc = {k: [] for k in evs}
                epochs = []
                for seed in range(a.seeds):
                    out, ep, net, (ymu, ysd) = fit_predict_seq(arch, Xtr, y[tr], dates[tr], Xev, seed, (fit_mask, val_mask), device=a.device)
                    if a.save_models:
                        mdir = OUT / "models"
                        mdir.mkdir(exist_ok=True)
                        tag = f"{fold.replace(':', '_')}__{fs}__{arch}__s{seed}"
                        torch.save(net.state_dict(), mdir / f"{tag}.pt")
                        np.savez(mdir / f"{tag}.npz", mu=pre.mu, sd=pre.sd, cols=pre.cols, ymu=ymu, ysd=ysd)
                    epochs.append(ep)
                    for k in evs:
                        acc[k].append(out[k])
                out = {k: np.mean(v, axis=0) for k, v in acc.items()}
                for split, idx in evs.items():
                    record(fold, split, fs, arch, df.loc[idx], out[split])
                msg = " | ".join(f"{split} RMSE {np.sqrt(np.mean((out[split] - y[idx]) ** 2)):.1f}" for split, idx in evs.items())
                print(f"[{time.time() - t0:6.0f}s] {fold:<18} {fs:<7} {arch:<5} {msg}   epochs {epochs}  ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(metrics).to_csv(OUT / f"metrics_{a.protocol}.csv", index=False)
        pd.concat(preds).to_csv(OUT / f"predictions_{a.protocol}.csv.gz", index=False)
    complete_run(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
