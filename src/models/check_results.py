"""Validate fold coverage before comparing saved experiments."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from paths import CITIES_CSV, ROOT


def expected_folds(protocol):
    if protocol == "chrono":
        return {"chrono"}
    if protocol == "loco":
        return {f"loco:{c}" for c in pd.read_csv(CITIES_CSV).city}
    if protocol == "season_cv":
        return {f"months{[k + 1, k + 5, k + 9]}" for k in range(4)}
    raise ValueError(protocol)


def coverage_errors(metrics, protocol, pairs=None, folds=None):
    expected = expected_folds(protocol) if folds is None else set(folds)
    o = metrics[(metrics.split == "test") & (metrics.subset == "has_lag1") & (metrics.level == "overall")]
    errors = []
    if o.empty:
        return ["No headline test metrics"]
    if o.duplicated(["fold", "feature_set", "model"]).any():
        errors.append("Duplicate headline fold/model/feature entries")
    pairs = pairs if pairs is not None else sorted(set(zip(o.feature_set, o.model)))
    for fs, model in pairs:
        rows = o[(o.feature_set == fs) & (o.model == model)]
        actual = set(rows.fold)
        if actual != expected:
            errors.append(f"{fs}/{model}: missing {sorted(expected - actual)}, unexpected {sorted(actual - expected)}")
        if rows[["RMSE", "MAE"]].isna().any().any() or (rows.n < 2).any():
            errors.append(f"{fs}/{model}: invalid metrics or insufficient scored rows")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, help="One saved run; default checks original experiment folders")
    args = ap.parse_args()
    directories = [args.results_dir] if args.results_dir else [ROOT / "results" / f for f in ("baselines", "sequence", "attention", "spatial")]
    failures = 0
    for directory in directories:
        files = sorted(directory.glob("metrics_*.csv"))
        if not files:
            print(f"FAIL {directory}: no metrics files")
            failures += 1
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        for file in files:
            protocol = file.stem.removeprefix("metrics_")
            pairs = None
            if manifest:
                a = manifest["arguments"]
                models = a.get("models", a.get("archs", a.get("variants", [])))
                pairs = [(fs, model) for fs in a["sets"] for model in models]
                if manifest["family"] == "baselines":
                    pairs += [("-", m) for m in ("Persistence(yesterday)", "Climatology(city-month)")]
            errors = coverage_errors(pd.read_csv(file), protocol, pairs, manifest["folds"] if manifest else None)
            if manifest and manifest["status"] != "complete":
                errors.append(f"Run status is {manifest['status']}")
            print(f"{'FAIL' if errors else 'PASS'} {file}")
            for error in errors:
                print("  " + error)
            failures += bool(errors)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
