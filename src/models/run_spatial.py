"""
Environment-only spatiotemporal attention model: 7-day windows of satellite + weather for a patch of neighbouring grid cells.

    python src/models/run_spatial.py --protocol chrono --save-models
    options: --variants S-Uniform S-TA S-SA S-TA+SA  --sets SW-w7 SWT-w7  --radius 1 --stride 3  --seeds 3  --threads 3

Variants (seq_models.SpatialNet):  S-Uniform = uniform weights over days and over cells (attention-ablated control)
                                   S-TA = learned temporal attention only     S-SA = learned spatial attention only
                                   S-TA+SA = both.
Reference for "does the neighbourhood help": the single-cell TA / Uniform models in results/attention/ (same inputs, centre cell only).
Same splits / metrics as the other runners. Writes results/spatial/*.
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
from seq_models import fit_predict_seq
from validation import temporal_validation_split
from spatial_data import SpatialPreprocessor, build_patches

warnings.filterwarnings("ignore")
SETS = {"SW-w7": False, "SWT-w7": True}          # name -> use_season


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["chrono", "loco", "season_cv"])
    ap.add_argument("--variants", nargs="+", default=["S-Uniform", "S-TA", "S-SA", "S-TA+SA"], choices=["S-Uniform", "S-TA", "S-SA", "S-TA+SA"])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="spatial")
    ap.add_argument("--save-models", action="store_true")
    add_run_args(ap)
    a = ap.parse_args()
    if getattr(a, "seeds", 1) < 1:
        ap.error("--seeds must be positive")
    if a.radius < 0 or a.stride < 1:
        ap.error("--radius must be nonnegative and --stride must be positive")
    torch.set_num_threads(a.threads)
    a.device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; install CUDA-enabled PyTorch or choose --device cpu")
    print(f"Training device: {a.device}" + (f" · {torch.cuda.get_device_name(0)}" if a.device == "cuda" else ""), flush=True)

    df = load_dataset()
    if a.quick:
        df = df.sample(4000, random_state=0).sort_values(["date", "station_id"]).reset_index(drop=True)
    P, present, offs = build_patches(df, a.radius, a.stride)
    dates = df.date.values
    y = df.y.values
    splits = select_splits(make_splits(df, a.protocol), a.folds)
    OUT = start_run(a, "spatial", df, splits)
    print(f"{len(df):,} rows | patch {2 * a.radius + 1}x{2 * a.radius + 1} stride {a.stride} | protocol {a.protocol} | "
          f"variants {a.variants} | sets {a.sets} | seeds {a.seeds}", flush=True)

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
            fit_mask, val_mask = temporal_validation_split(dates[tr], P.shape[2])
            pre = SpatialPreprocessor(SETS[fs]).fit(P[tr[fit_mask]], present[tr[fit_mask]])
            Xtr = pre.transform(P[tr], present[tr], offs)
            Xev = {k: pre.transform(P[i], present[i], offs) for k, i in evs.items()}
            for var in a.variants:
                t1 = time.time()
                acc = {k: [] for k in evs}
                epochs = []
                for seed in range(a.seeds):
                    out, ep, net, (ymu, ysd) = fit_predict_seq(var, Xtr, y[tr], dates[tr], Xev, seed, (fit_mask, val_mask), device=a.device)
                    if a.save_models:
                        mdir = OUT / "models"
                        mdir.mkdir(exist_ok=True)
                        tag = f"{fold.replace(':', '_')}__{fs}__{var}__s{seed}"
                        torch.save(net.state_dict(), mdir / f"{tag}.pt")
                        np.savez(mdir / f"{tag}.npz", mu=pre.mu, sd=pre.sd, cols=pre.cols, ymu=ymu, ysd=ysd, radius=a.radius, stride=a.stride)
                    epochs.append(ep)
                    for k in evs:
                        acc[k].append(out[k])
                out = {k: np.mean(v, axis=0) for k, v in acc.items()}
                for split, idx in evs.items():
                    record(fold, split, fs, var, df.loc[idx], out[split])
                msg = " | ".join(f"{split} RMSE {np.sqrt(np.mean((out[split] - y[idx]) ** 2)):.1f}" for split, idx in evs.items())
                print(f"[{time.time() - t0:6.0f}s] {fold:<18} {fs:<7} {var:<10} {msg}   epochs {epochs}  ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(metrics).to_csv(OUT / f"metrics_{a.protocol}.csv", index=False)
        pd.concat(preds).to_csv(OUT / f"predictions_{a.protocol}.csv.gz", index=False)
    complete_run(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
