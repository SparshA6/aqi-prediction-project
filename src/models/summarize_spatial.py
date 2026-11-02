"""Spatiotemporal (neighbouring-cell patch) model summary -> docs/SPATIAL_MODEL_RESULTS.md

Part 1: accuracy of the patch models vs the attention-ablated patch control (S-Uniform) and vs the single-cell models
        with the same inputs (results/attention Uniform / TA, results/sequence GRU).
Part 2: what the learned spatial / temporal attention looks like (chrono test rows, saved models, 3 seeds). Descriptive only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from data import load_dataset, make_splits
from paths import ROOT
from seq_data import STEP_FEATURES
from seq_models import make_net
from spatial_data import SpatialPreprocessor, build_patches, patch_offsets
from summarize_baselines import md
from check_results import coverage_errors

SUB = "has_lag1"
SETS = {"SW-w7": False, "SWT-w7": True}
PATCH = ["S-Uniform", "S-TA", "S-SA", "S-TA+SA"]
SINGLE = [("GRU", "sequence"), ("Uniform", "attention"), ("TA", "attention")]
SEEDS = 3
RADIUS, STRIDE = 1, 3


def overall(m):
    return m[(m.split == "test") & (m.subset == SUB) & (m.level == "overall")]


def accuracy_tables():
    res = {}
    notices = []
    for p in ("chrono", "loco", "season_cv"):
        frames = []
        for d in ("spatial", "attention", "sequence"):
            f = ROOT / "results" / d / f"metrics_{p}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
        if not (ROOT / "results" / "spatial" / f"metrics_{p}.csv").exists():
            continue
        m = pd.concat(frames)
        relevant = m[m.feature_set.isin(SETS) & m.model.isin(["GRU", "Uniform", "TA"] + PATCH)]
        errors = coverage_errors(relevant, p, [(fs, model) for fs in SETS for model in ["GRU", "Uniform", "TA"] + PATCH])
        if errors:
            notices += [f"**{p} comparison omitted because fold coverage is incomplete.**", ""]
            notices += ["- " + error for error in errors] + [""]
            continue
        g = overall(relevant).groupby(["feature_set", "model"]).agg(RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
        res[p] = g.set_index(["feature_set", "model"])
    if not res:
        return notices
    label = {"GRU": "single cell: GRU (last hidden)", "Uniform": "single cell: Uniform (attention ablated)", "TA": "single cell: TA",
             "S-Uniform": "patch: S-Uniform (attention ablated)", "S-TA": "patch: S-TA", "S-SA": "patch: S-SA", "S-TA+SA": "patch: S-TA+SA"}
    rows = []
    for fs in SETS:
        for mod in ["GRU", "Uniform", "TA"] + PATCH:
            r = {"input set": fs, "model": label[mod]}
            for p, g in res.items():
                if (fs, mod) in g.index:
                    r[f"{p} RMSE"], r[f"{p} R2"] = g.loc[(fs, mod), "RMSE"], g.loc[(fs, mod), "R2"]
            rows.append(r)
    out = notices + ["## 1. Accuracy: does the neighbourhood help, and does learned attention help?", "",
           "Test RMSE / R2 (chrono: Nov-Dec; loco: mean of 8 held-out cities; season_cv: mean of 4 folds). `patch` = 3x3 neighbouring grid cells "
           "(stride 3 = ~0.09 deg, ~10 km) each with a 7-day window; `S-Uniform` = same network with uniform weights over days and cells; "
           "`S-TA` / `S-SA` = learned temporal / spatial attention only; `single cell` rows = the models from the previous stage (centre cell only, same inputs).", "",
           md(pd.DataFrame(rows), 2), ""]
    return out


def profiles():
    mdir = ROOT / "results" / "spatial" / "models"
    if not mdir.exists():
        return []
    df = load_dataset()
    P, present, offs = build_patches(df, RADIUS, STRIDE)
    fold, tr, evs = next(make_splits(df, "chrono"))
    te = evs["test"]
    offsets = patch_offsets(RADIUS, STRIDE)
    C = len(offsets)
    full = present[te].all(1)                                    # rows whose whole 3x3 patch exists (no masking confound)
    out = ["## 2. What the learned attention looks like (chrono test rows, 3 seeds)", "",
           f"Descriptive only. Spatial weights are averaged over the {int(full.sum()):,} test rows whose full 3x3 patch exists (no masking); uniform = 1/9 = 0.111. "
           "Grid layout: rows = north to south of the station cell... printed as dy (rows) x dx (columns); the middle cell is the station's own cell. "
           "dy>0 = larger latitude index (north), dx>0 = east. Days are counted back from the target day (d-0 = same day).", ""]
    torch.set_num_threads(2)
    for fs, season in SETS.items():
        for var in ("S-SA", "S-TA", "S-TA+SA"):
            gam, alp = [], []
            for s in range(SEEDS):
                f = mdir / f"chrono__{fs}__{var}__s{s}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                pre = SpatialPreprocessor(season)
                pre.mu, pre.sd = z["mu"], z["sd"]
                Xte = torch.from_numpy(pre.transform(P[te], present[te], offs))
                net = make_net(var, Xte.shape[-1])
                if int(z["radius"]) != RADIUS or int(z["stride"]) != STRIDE:
                    raise ValueError("Saved patch geometry differs from the summary configuration")
                net.load_state_dict(torch.load(f.with_suffix(".pt"), map_location="cpu", weights_only=True))
                net.eval()
                aa, gg = [], []
                with torch.no_grad():
                    for start in range(0, len(Xte), 256):
                        _, a, g = net(Xte[start:start + 256], return_attn=True)
                        aa.append(a.numpy())
                        gg.append(g.numpy())
                gam.append(np.concatenate(gg))
                alp.append(np.concatenate(aa))
            if not gam:
                continue
            out += [f"### {fs} / {var}", ""]
            if var != "S-TA":
                G = np.stack(gam)[:, full]                       # (seeds, n_full, C)
                mg = G.mean(axis=(0, 1))
                grid = pd.DataFrame(mg.reshape(3, 3)[::-1], index=["dy=+1 (N)", "dy=0", "dy=-1 (S)"], columns=["dx=-1 (W)", "dx=0", "dx=+1 (E)"])
                ent = -(G * np.log(G + 1e-12)).sum(-1) / np.log(C)
                per_seed = G.mean(1)
                agree = np.mean([np.corrcoef(per_seed[i], per_seed[j])[0, 1] for i in range(len(G)) for j in range(i + 1, len(G))]) if len(G) > 1 else np.nan
                out += ["Spatial attention (mean weight per cell):", "", md(grid.reset_index().rename(columns={"index": ""}), 3), "",
                        f"centre-cell weight {mg[(C - 1) // 2]:.3f} (uniform 0.111) | normalised entropy {ent.mean():.3f} | row-to-row std of the centre weight "
                        f"{G[..., (C - 1) // 2].std():.3f} | seed-to-seed correlation of the 9-cell profile {agree:.2f}", ""]
            if var != "S-SA":
                A = np.stack(alp)[:, :, (C - 1) // 2, :]         # centre cell's temporal weights (seeds, n, T)
                ma = A.mean(axis=(0, 1))
                out += ["Temporal attention of the centre cell (mean weight per day; uniform 0.143):", "",
                        md(pd.DataFrame([{f"d-{6 - k}": ma[k] for k in range(A.shape[-1])}]), 3), ""]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accuracy-only", action="store_true", help="Skip saved-model weight profiles")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "SPATIAL_MODEL_RESULTS.md")
    args = ap.parse_args()
    out = ["# Spatiotemporal (neighbouring-cell patch) model results", "",
           "Auto-generated by `src/models/summarize_spatial.py`. Environment-only inputs (satellite + weather, no AQI history, no absolute lat/lon; only each cell's offset from the "
           "station). 3 seeds per cell, fixed untuned hyper-parameters.", ""]
    out += accuracy_tables()
    if not args.accuracy_only:
        out += profiles()
    args.output.write_text("\n".join(out), encoding="utf-8")
    print(f"Summary saved: {args.output}")


if __name__ == "__main__":
    main()
