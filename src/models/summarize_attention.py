"""Attention-model summary -> docs/ATTENTION_MODEL_RESULTS.md

Part 1: accuracy of the attention variants vs their attention-ablated control (Uniform) and the plain GRU, three protocols.
Part 2: what the learned attention looks like on the chronological test rows (saved models, 3 seeds):
        which days get weight, which input features get weight, how concentrated (normalised entropy) and whether
        seeds agree. This is a DESCRIPTION of the weights only -- whether they are faithful is tested in the XAI stage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from data import SAT, load_dataset, make_splits
from paths import ROOT
from seq_data import STEP_FEATURES, build_windows
from seq_models import make_net
from summarize_baselines import md

SUB = "has_lag1"
VARIANTS = ["Uniform", "TA", "FA", "TA+FA"]
SETS = {"SW-w7": (7, False), "SWT-w7": (7, True)}
SEEDS = 3


def overall(m):
    return m[(m.split == "test") & (m.subset == SUB) & (m.level == "overall")]


def accuracy_tables():
    out = []
    res = {}
    for p in ("chrono", "loco", "season_cv"):
        fa, fg = ROOT / "results" / "attention" / f"metrics_{p}.csv", ROOT / "results" / "sequence" / f"metrics_{p}.csv"
        if not fa.exists():
            continue
        m = pd.concat([pd.read_csv(fa), pd.read_csv(fg)])
        m = m[m.model.isin(VARIANTS + ["GRU"]) & m.feature_set.isin(SETS)]
        g = overall(m).groupby(["feature_set", "model"]).agg(RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
        res[p] = g.set_index(["feature_set", "model"])
    if not res:
        return out
    rows = []
    for fs in SETS:
        for mod in ["GRU"] + VARIANTS:
            r = {"input set": fs, "model": mod + (" (last hidden, no attention)" if mod == "GRU" else " (attention ablated)" if mod == "Uniform" else "")}
            for p, g in res.items():
                if (fs, mod) in g.index:
                    r[f"{p} RMSE"], r[f"{p} R2"] = g.loc[(fs, mod), "RMSE"], g.loc[(fs, mod), "R2"]
            rows.append(r)
    out += ["## 1. Accuracy: attention variants vs the attention-ablated control", "",
            "Test RMSE / R2 (chrono: Nov-Dec; loco: mean of 8 held-out cities; season_cv: mean of 4 folds). `Uniform` = same GRU network with fixed uniform weights over days "
            "(mean-pool), i.e. the control that isolates what *learning* the attention adds. `TA` = temporal attention, `FA` = feature attention, `TA+FA` = both.", "",
            md(pd.DataFrame(rows), 2), ""]
    fc = ROOT / "results" / "attention" / "metrics_chrono.csv"
    fg = ROOT / "results" / "sequence" / "metrics_chrono.csv"
    m = pd.concat([pd.read_csv(fc), pd.read_csv(fg)])
    e = m[(m.split == "test") & (m.subset == SUB) & (m.level == "episode") & (m.group == "AQI >= 300") & m.model.isin(VARIANTS + ["GRU"]) & m.feature_set.isin(SETS)]
    out += ["**Chrono test, days with true AQI >= 300**: RMSE / bias", "", md(e[["feature_set", "model", "n", "RMSE", "bias"]].sort_values(["feature_set", "model"]), 1), ""]
    return out


def attention_profiles():
    mdir = ROOT / "results" / "attention" / "models"
    if not mdir.exists():
        return []
    df = load_dataset()
    X = build_windows(df)
    fold, tr, evs = next(make_splits(df, "chrono"))
    te = evs["test"]
    out = ["## 2. What the learned attention looks like (chrono test rows, 3 seeds)", "",
           "Descriptive only. Days are counted back from the target day (d-0 = same day). Feature weights are the gate `beta` averaged over the 7 days and all test rows, "
           "normalised so a uniform gate = 1/F. Entropy is normalised to [0,1] (1 = perfectly uniform, i.e. no selection).", ""]
    for fs, (window, use_season) in SETS.items():
        for var in ("TA", "FA", "TA+FA"):
            alphas, betas = [], []
            names = None
            for s in range(SEEDS):
                tag = f"chrono__{fs}__{var}__s{s}"
                f = mdir / f"{tag}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                cols = list(z["cols"])
                names = [STEP_FEATURES[c] for c in cols] + [f"{b}_missing" for b in SAT]
                Xw = X[te][:, -window:, :][:, :, cols]
                miss = np.isnan(Xw[:, :, :len(SAT)]).astype(np.float32)
                Z = np.nan_to_num((Xw - z["mu"]) / z["sd"], nan=0.0)
                inp = torch.from_numpy(np.concatenate([Z, miss], axis=2).astype(np.float32))
                net = make_net(var, inp.shape[2])
                net.load_state_dict(torch.load(f.with_suffix(".pt")))
                net.eval()
                with torch.no_grad():
                    _, a, b = net(inp, return_attn=True)
                alphas.append(a.numpy())
                betas.append(None if b is None else b.numpy())
            if not alphas:
                continue
            out += [f"### {fs} / {var}", ""]
            if var != "FA":
                A = np.stack(alphas)                                          # (seeds, n, T)
                mean_a = A.mean(axis=(0, 1))
                ent = -(A * np.log(A + 1e-12)).sum(-1) / np.log(A.shape[-1])
                per_seed = A.mean(1)
                agree = np.mean([np.corrcoef(per_seed[i], per_seed[j])[0, 1] for i in range(len(A)) for j in range(i + 1, len(A))]) if len(A) > 1 else np.nan
                out += ["Temporal attention (mean weight per day; uniform would be %.3f):" % (1 / A.shape[-1]), "",
                        md(pd.DataFrame([{f"d-{window - 1 - k}": mean_a[k] for k in range(window)}]), 3), "",
                        f"normalised entropy {ent.mean():.3f} | row-to-row std of the weight on the most-weighted day {A[..., mean_a.argmax()].std():.3f} | "
                        f"seed-to-seed correlation of the mean profile {agree:.2f}", ""]
            if var != "TA":
                B = np.stack([b for b in betas if b is not None])            # (seeds, n, T, F)
                Bm = B.mean(axis=(1, 2))                                     # (seeds, F)
                mean_b = Bm.mean(0)
                sd_b = Bm.std(0)
                ent = -(B * np.log(B + 1e-12)).sum(-1) / np.log(B.shape[-1])
                agree = np.mean([np.corrcoef(Bm[i], Bm[j])[0, 1] for i in range(len(Bm)) for j in range(i + 1, len(Bm))]) if len(Bm) > 1 else np.nan
                tab = pd.DataFrame({"feature": names, "mean gate": mean_b, "seed sd": sd_b, "vs uniform": mean_b * len(names)}).sort_values("mean gate", ascending=False)
                out += [f"Feature attention (uniform would be {1 / len(names):.3f}):", "", md(tab, 3), "",
                        f"normalised entropy {ent.mean():.3f} | seed-to-seed correlation of the mean profile {agree:.2f}", ""]
    return out


def main():
    out = ["# Attention model results (environment-only)", "",
           "Auto-generated by `src/models/summarize_attention.py`. Inputs: satellite + weather over a 7-day window, no AQI history. "
           "3 seeds per cell, fixed untuned hyper-parameters, early stopping on held-out weeks of the training rows.", ""]
    out += accuracy_tables()
    out += attention_profiles()
    (ROOT / "docs" / "ATTENTION_MODEL_RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/ATTENTION_MODEL_RESULTS.md")


if __name__ == "__main__":
    main()
