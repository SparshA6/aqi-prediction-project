"""Exploratory temporal-attention audit of saved chronological checkpoints.

Replaces selected input days with a training-mean step and compares attention
ranking against random/reverse rankings and leave-one-day-out sensitivity.
Attention pools recurrent hidden states, so these are operational tests of input
day sensitivity, not a one-to-one attribution of a hidden state's weight.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from data import load_dataset, make_splits
from paths import ALIGNED_CSV, ROOT
from seq_data import Preprocessor, build_windows
from seq_models import make_net


def forward(net, x, batch=256):
    predictions, attention = [], []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            y, alpha, _ = net(torch.from_numpy(x[start:start + batch]), return_attn=True)
            predictions.append(y.numpy())
            attention.append(alpha.numpy())
    return np.concatenate(predictions), np.concatenate(attention)


def replace_days(x, order, k, replacement):
    changed = x.copy()
    rows = np.arange(len(x))[:, None]
    changed[rows, order[:, :k]] = replacement
    return changed


def rank_correlation(a, b):
    # Average ranks handle constant or tied attention/sensitivity honestly.
    ar = pd.DataFrame(a).rank(axis=1).to_numpy()
    br = pd.DataFrame(b).rank(axis=1).to_numpy()
    ar -= ar.mean(axis=1, keepdims=True)
    br -= br.mean(axis=1, keepdims=True)
    denominator = np.sqrt((ar * ar).sum(1) * (br * br).sum(1))
    return np.divide((ar * br).sum(1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["TA", "TA+FA"], default="TA")
    ap.add_argument("--set", dest="feature_set", choices=["SW-w7", "SWT-w7"], default="SWT-w7")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "results" / "attention" / "models")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--random-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    if args.sample < 2 or args.random_repeats < 1:
        ap.error("--sample must be at least 2 and --random-repeats positive")
    torch.set_num_threads(args.threads)
    checkpoints = sorted(args.models_dir.glob(f"chrono__{args.feature_set}__{args.model}__s*.npz"))
    if not checkpoints:
        raise SystemExit("No matching saved preprocessing/checkpoint files")
    run_manifest = args.models_dir.parent / "manifest.json"
    if run_manifest.exists():
        manifest = json.loads(run_manifest.read_text())
        if manifest["arguments"].get("quick"):
            raise SystemExit("Audit requires full-run checkpoints; smoke-test sampling is a different experiment")
        if manifest.get("dataset_sha256") != sha256(ALIGNED_CSV.read_bytes()).hexdigest():
            raise SystemExit("Saved checkpoint dataset fingerprint differs from the current dataset")
    for file in checkpoints:
        if not file.with_suffix(".pt").exists():
            raise SystemExit(f"Missing weights: {file.with_suffix('.pt')}")
    df = load_dataset()
    x = build_windows(df)
    _, tr, splits = next(make_splits(df, "chrono"))
    rng = np.random.default_rng(args.seed)
    te = np.sort(rng.choice(splits["test"], min(args.sample, len(splits["test"])), replace=False))
    # Order controls are shared across seed models for paired comparisons.
    random_orders = [np.argsort(rng.random((len(te), 7)), axis=1) for _ in range(args.random_repeats)]
    base_predictions, alphas, sensitivities, curves = [], [], [], []
    for file in checkpoints:
        z = np.load(file)
        pre = Preprocessor(args.feature_set.startswith("SWT"))
        pre.mu, pre.sd, pre.cols = z["mu"], z["sd"], z["cols"].tolist()
        xt = pre.transform(x[te], 7)
        replacement = pre.transform(x[tr], 7).mean(axis=(0, 1))
        net = make_net(args.model, xt.shape[-1])
        net.load_state_dict(torch.load(file.with_suffix(".pt"), map_location="cpu", weights_only=True))
        base, alpha = forward(net, xt)
        scale, offset = float(z["ysd"]), float(z["ymu"])
        base_predictions.append(base * scale + offset)
        alphas.append(alpha)
        sensitivity = np.empty((len(te), 7))
        for day in range(7):
            changed = xt.copy()
            changed[:, day] = replacement
            y, _ = forward(net, changed)
            sensitivity[:, day] = np.abs(y - base) * scale
        sensitivities.append(sensitivity)
        rankings = {"attention": [np.argsort(-alpha, axis=1)],
                    "reverse_attention": [np.argsort(alpha, axis=1)], "random": random_orders}
        for ranking, orders in rankings.items():
            for k in (1, 2, 3, 5, 7):
                changes = []
                for order in orders:
                    changed = replace_days(xt, order, k, replacement)
                    y, _ = forward(net, changed)
                    changes.append(np.abs(y - base) * scale)
                for row, change in zip(te, np.mean(changes, axis=0)):
                    curves.append(dict(checkpoint=file.stem, row=int(row), ranking=ranking, days_replaced=k, prediction_change=float(change)))
    out = args.output_dir or ROOT / "results" / "faithfulness" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    alpha = np.mean(alphas, axis=0)
    sensitivity = np.mean(sensitivities, axis=0)
    rows = df.loc[te, ["station_id", "city", "date", "y"]].copy()
    rows["row"] = te
    rows["pred"] = np.mean(base_predictions, axis=0)
    # Correlations are computed within each seed model, then averaged per row.
    correlations = np.stack([rank_correlation(a, s) for a, s in zip(alphas, sensitivities)])
    rows["attention_sensitivity_spearman"] = pd.DataFrame(correlations).mean(axis=0).to_numpy()
    for t in range(7):
        rows[f"attention_d{6-t}"] = alpha[:, t]
        rows[f"sensitivity_d{6-t}"] = sensitivity[:, t]
    rows.to_csv(out / "row_audit.csv", index=False)
    c = pd.DataFrame(curves)
    c.to_csv(out / "deletion_rows.csv.gz", index=False)
    summary = c.groupby(["ranking", "days_replaced"]).prediction_change.mean().unstack(0)
    summary["attention_minus_random"] = summary.attention - summary.random
    summary["attention_minus_reverse"] = summary.attention - summary.reverse_attention
    # Paired gaps averaged across seed models; resample calendar dates as clusters.
    paired = c.pivot_table(index=["row", "days_replaced"], columns="ranking", values="prediction_change").reset_index()
    paired["date"] = paired.row.map(df.date)
    boot_rng = np.random.default_rng(args.seed)
    for k in summary.index:
        selected = paired[paired.days_replaced == k].copy()
        for control in ("random", "reverse_attention"):
            selected["gap"] = selected.attention - selected[control]
            grouped = selected.groupby("date").gap.agg(["sum", "count"])
            draws = boot_rng.integers(0, len(grouped), size=(1000, len(grouped)))
            means = grouped["sum"].to_numpy()[draws].sum(1) / grouped["count"].to_numpy()[draws].sum(1)
            label = "random" if control == "random" else "reverse"
            summary.loc[k, f"gap_vs_{label}_ci_low"] = np.quantile(means, 0.025)
            summary.loc[k, f"gap_vs_{label}_ci_high"] = np.quantile(means, 0.975)
    summary.to_csv(out / "deletion_summary.csv")
    rows["regime"] = np.where(rows.y >= 300, "AQI >= 300", "AQI < 300")
    strata = []
    for level in ("city", "regime"):
        for group, sub in rows.groupby(level):
            strata.append(dict(level=level, group=group, n=len(sub),
                               mean_spearman=sub.attention_sensitivity_spearman.mean(),
                               RMSE=float(np.sqrt(np.mean((sub.pred - sub.y) ** 2)))))
    pd.DataFrame(strata).to_csv(out / "stratified_summary.csv", index=False)
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "manifest.json").write_text(json.dumps(dict(arguments=args_dict, checkpoints=[str(f) for f in checkpoints],
        dataset_sha256=sha256(ALIGNED_CSV.read_bytes()).hexdigest(),
        checkpoint_sha256={str(f): sha256(f.read_bytes()).hexdigest() for p in checkpoints for f in (p, p.with_suffix(".pt"))},
        checkpoint_provenance="run manifest checked" if run_manifest.exists() else "legacy checkpoints have no run manifest",
        sampled_rows=len(te), mean_within_seed_spearman=float(rows.attention_sensitivity_spearman.mean()),
        perturbation="replace whole input day with mean transformed training step",
        limitations=["Exploratory subset; not a full faithfulness conclusion", "Mean replacement can be out of distribution",
                     "Recurrent hidden-state attention is not direct input-day attribution",
                     "Checkpoint training procedure may predate purged validation",
                     "Paired gap intervals resample dates only; not independent stations or full training uncertainty",
                     "No feature or spatial faithfulness audit yet"]), indent=2), encoding="utf-8")
    print(f"Audit saved: {out}")
    print(f"Mean within-seed attention/sensitivity Spearman: {rows.attention_sensitivity_spearman.mean():.3f}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
