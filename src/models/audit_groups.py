"""Feature or spatial-cell faithfulness audit with SHAP and paired controls."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch import nn

from audit_attention import rank_correlation
from data import load_dataset, make_splits
from paths import ALIGNED_CSV
from seq_data import Preprocessor, build_windows
from seq_models import make_net
from spatial_data import SpatialPreprocessor, build_patches


class OutputColumn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(-1)


def groups_for(n, seasonal):
    groups = {"NO2": [0, n - 3], "HCHO": [1, n - 2], "AOD": [2, n - 1],
              "temperature": [3], "humidity": [4], "wind_speed": [5],
              "wind_direction": [6, 7], "pressure": [8], "precipitation": [9]}
    if seasonal:
        groups["season"] = [10, 11]
    return groups


def predict(net, x):
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(x[i:i+128])).numpy() for i in range(0, len(x), 128)])


def perturb(x, replacement, order, k, groups, spatial):
    changed = x.copy()
    for row in range(len(x)):
        for group in order[row, :k]:
            if spatial:
                # Replace environmental channels only, preserving offsets and presence mask.
                changed[row, group, :, :-3] = replacement[row, group, :, :-3]
            else:
                cols = list(groups.values())[group]
                changed[row][:, cols] = replacement[row][:, cols]
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["feature", "spatial"], default="feature")
    ap.add_argument("--model", default="FA")
    ap.add_argument("--set", dest="feature_set", choices=["SW-w7", "SWT-w7"], default="SWT-w7")
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--background", type=int, default=30)
    ap.add_argument("--shap-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if min(args.sample, args.background, args.shap_samples) < 2:
        ap.error("Sampling parameters must be at least two")
    spatial = args.kind == "spatial"
    valid = ["S-SA", "S-TA+SA"] if spatial else ["FA", "TA+FA"]
    if args.model not in valid:
        ap.error(f"This audit requires one of {valid}")
    import shap
    torch.set_num_threads(2)
    family = "spatial" if spatial else "attention"
    mdir = args.models_dir or ROOT / "results" / family / "models"
    files = sorted(mdir.glob(f"chrono__{args.feature_set}__{args.model}__s*.npz"))
    if not files:
        raise SystemExit("No matching chronological checkpoints")
    manifest_path = mdir.parent / "manifest.json"
    checkpoint_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if checkpoint_manifest:
        if checkpoint_manifest["arguments"].get("quick"):
            raise SystemExit("Smoke checkpoints cannot be used for research audits")
        if checkpoint_manifest.get("dataset_sha256") != sha256(ALIGNED_CSV.read_bytes()).hexdigest():
            raise SystemExit("Current data differs from checkpoint training data")
    df = load_dataset()
    _, tr, ev = next(make_splits(df, "chrono"))
    rng = np.random.default_rng(args.seed)
    raw = build_windows(df) if not spatial else None
    if spatial:
        z0 = np.load(files[0])
        raw, present, offsets = build_patches(df, int(z0["radius"]), int(z0["stride"]))
        candidates = ev["test"][present[ev["test"]].all(axis=1)]
        donors = tr[present[tr].all(axis=1)]
    else:
        candidates, donors = ev["test"], tr
    if len(candidates) < 2 or len(donors) < 2:
        raise SystemExit("Not enough full-patch rows for matched spatial perturbation")
    te = np.sort(rng.choice(candidates, min(args.sample, len(candidates)), replace=False))
    bg = rng.choice(donors, min(args.background, len(donors)), replace=False)
    # Select same-city donor rows closest in cyclical seasonal date where available.
    donor_rows = []
    for row in te:
        pool = donors[df.loc[donors, "city"].to_numpy() == df.loc[row, "city"]]
        if not len(pool):
            pool = donors
        distance = np.abs(df.loc[pool, "date"].dt.dayofyear.to_numpy() - df.loc[row, "date"].dayofyear)
        distance = np.minimum(distance, 366 - distance)
        nearest = pool[np.argsort(distance)[:min(30, len(pool))]]
        donor_rows.append(rng.choice(nearest))
    curves, correlations, raw_rows = [], [], []
    for file in files:
        z = np.load(file)
        pre = SpatialPreprocessor(args.feature_set.startswith("SWT")) if spatial else Preprocessor(args.feature_set.startswith("SWT"))
        pre.mu, pre.sd, pre.cols = z["mu"], z["sd"], z["cols"].tolist()
        def transform(idx):
            return pre.transform(raw[idx], present[idx], offsets) if spatial else pre.transform(raw[idx], 7)
        xt, xb, xd = transform(te), transform(bg), transform(np.asarray(donor_rows))
        net = make_net(args.model, xt.shape[-1]).eval()
        net.load_state_dict(torch.load(file.with_suffix(".pt"), weights_only=True, map_location="cpu"))
        with torch.no_grad():
            base, alpha, beta = net(torch.from_numpy(xt), return_attn=True)
        base = base.numpy()
        scale = float(z["ysd"])
        groups = {f"cell_{i}": [i] for i in range(xt.shape[1])} if spatial else groups_for(xt.shape[-1], args.feature_set.startswith("SWT"))
        if spatial:
            weight = beta.numpy()
        else:
            gate = beta.numpy().mean(1)
            weight = np.column_stack([gate[:, cols].sum(1) for cols in groups.values()])
        values = shap.GradientExplainer(OutputColumn(net), torch.from_numpy(xb), batch_size=32).shap_values(torch.from_numpy(xt), nsamples=args.shap_samples, rseed=args.seed)
        values = np.asarray(values)
        if values.ndim == xt.ndim + 1:
            values = values[..., 0]
        if spatial:
            shap_score = np.abs(values[..., :-3]).sum(axis=(2, 3))
        else:
            shap_score = np.column_stack([np.abs(values[:, :, cols]).sum(axis=(1, 2)) for cols in groups.values()])
        replacements = {"mean": np.broadcast_to(xb.mean(axis=0), xt.shape).copy(), "seasonal_donor": xd}
        for baseline, replacement in replacements.items():
            sensitivity = np.empty_like(weight)
            for g in range(len(groups)):
                order = np.full((len(te), 1), g)
                sensitivity[:, g] = np.abs(predict(net, perturb(xt, replacement, order, 1, groups, spatial)) - base) * scale
            for method, score in [("attention", weight), ("SHAP", shap_score)]:
                corr = rank_correlation(score, sensitivity)
                for row, value in zip(te, corr):
                    correlations.append(dict(row=int(row), city=df.loc[row, "city"], y=float(df.loc[row, "y"]), checkpoint=file.stem, baseline=baseline, method=method, spearman=float(value)))
            orders = {"attention": [np.argsort(-weight, axis=1)], "SHAP": [np.argsort(-shap_score, axis=1)],
                      "reverse_attention": [np.argsort(weight, axis=1)],
                      "random": [np.argsort(rng.random(weight.shape), axis=1) for _ in range(10)]}
            for method, ranks in orders.items():
                for k in sorted({1, 2, len(groups)//2, len(groups)}):
                    change = np.mean([np.abs(predict(net, perturb(xt, replacement, order, k, groups, spatial)) - base) * scale for order in ranks], axis=0)
                    for row, value in zip(te, change):
                        curves.append(dict(row=int(row), checkpoint=file.stem, baseline=baseline, method=method, groups_replaced=k, change=float(value)))
            for i, row in enumerate(te):
                for g, label in enumerate(groups):
                    raw_rows.append(dict(row=int(row), checkpoint=file.stem, baseline=baseline, group=label,
                                         attention=float(weight[i,g]), SHAP=float(shap_score[i,g]*scale), sensitivity=float(sensitivity[i,g])))
    out = ROOT / "results" / "faithfulness" / (args.kind + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    pd.DataFrame(raw_rows).to_csv(out / "group_rows.csv.gz", index=False)
    corr = pd.DataFrame(correlations)
    corr.to_csv(out / "correlations.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "deletion_rows.csv.gz", index=False)
    summary = pd.DataFrame(curves).groupby(["baseline", "method", "groups_replaced"]).change.mean().reset_index()
    summary.to_csv(out / "deletion_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(kind=args.kind, model=args.model, feature_set=args.feature_set,
        models_dir=str(mdir), dataset_sha256=sha256(ALIGNED_CSV.read_bytes()).hexdigest(),
        checkpoint_sha256={str(f): sha256(f.read_bytes()).hexdigest() for p in files for f in (p,p.with_suffix(".pt"))},
        validation="purged date validation" if checkpoint_manifest else "legacy checkpoints; earlier validation procedure",
        rows=len(te), seeds=len(files), shap_version=shap.__version__, shap_samples=args.shap_samples,
        scope="Exploratory; spatial sample restricted to full patches", replacement_controls=list(replacements),
        donor_selection="same city, nearest cyclical seasonal dates within training period",
        limitations=["No causal conclusion", "Mean masks may be out of distribution", "Seasonal donors can still be mismatched",
                     "Small sampled subset", "Absolute SHAP aggregates are group importance proxies, not causal effects"]), indent=2), encoding="utf-8")
    print(f"Group audit saved: {out}")
    print(corr.groupby(["baseline", "method"]).spearman.mean().round(3).to_string())


if __name__ == "__main__":
    main()
