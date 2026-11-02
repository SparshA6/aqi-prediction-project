"""Attention versus uniform control on both input sets and all protocols, from saved predictions; no retraining.

Gain = control RMSE minus attention RMSE, averaged over folds (one fold for chrono). Intervals are paired
calendar-week block bootstraps. They cover test-week sampling only, not training-seed variation, and are not
adjusted for the number of comparisons.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/attention_stability"
RUNS = {("attention", "chrono"): "corrected-v2-attention-chrono", ("spatial", "chrono"): "corrected-v2-spatial-chrono",
        ("attention", "loco"): "corrected-v2-gpu-attention-loco", ("spatial", "loco"): "corrected-v2-gpu-spatial-loco",
        ("attention", "season_cv"): "corrected-v2-gpu-attention-season_cv", ("spatial", "season_cv"): "corrected-v2-gpu-spatial-season_cv"}
CONTROL = {"attention": "Uniform", "spatial": "S-Uniform"}


def fold_gain(totals):
    """totals: (folds, 3) of attention SSE, control SSE, rows. Folds absent from a resample are skipped."""
    present = totals[:, 2] > 0
    return np.mean(np.sqrt(totals[present, 1] / totals[present, 2]) - np.sqrt(totals[present, 0] / totals[present, 2]))


def compare(rows, control, repetitions=2000):
    q = rows.merge(control[["fold", "station_id", "date", "pred"]], on=["fold", "station_id", "date"], suffixes=("", "_control"), validate="one_to_one")
    if len(q) != len(rows) or len(q) != len(control):
        raise ValueError("Attention and control rows differ")
    dates = pd.to_datetime(q.date)
    q["week"] = (dates - dates.min()).dt.days // 7
    weeks = sorted(q.week.unique())
    q = q.assign(attention=(q.pred - q.y) ** 2, reference=(q.pred_control - q.y) ** 2, n=1)
    table = np.stack([g.groupby("week")[["attention", "reference", "n"]].sum().reindex(weeks, fill_value=0).to_numpy() for _, g in q.groupby("fold")])
    rng = np.random.default_rng(0)
    draws = [fold_gain(table[:, rng.integers(0, len(weeks), len(weeks))].sum(axis=1)) for _ in range(repetitions)]
    totals = table.sum(axis=1)
    per_fold = np.sqrt(totals[:, 1] / totals[:, 2]) - np.sqrt(totals[:, 0] / totals[:, 2])
    return dict(control_rmse=float(np.sqrt(totals[:, 1] / totals[:, 2]).mean()), attention_rmse=float(np.sqrt(totals[:, 0] / totals[:, 2]).mean()),
                gain=float(fold_gain(totals)), lower=float(np.quantile(draws, .025)), upper=float(np.quantile(draws, .975)),
                folds_improved=int((per_fold > 0).sum()), folds=len(per_fold))


def main():
    results = []
    for (family, protocol), run in RUNS.items():
        d = pd.read_csv(ROOT / "results" / family / "runs" / run / f"predictions_{protocol}.csv.gz")
        d = d[d.split == "test"]
        for feature_set, g in d.groupby("feature_set"):
            control = g[g.model == CONTROL[family]]
            for model in sorted(set(g.model) - {CONTROL[family]}):
                results.append(dict(protocol=protocol, family=family, feature_set=feature_set, model=model, **compare(g[g.model == model], control)))
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(OUT / "gains.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
