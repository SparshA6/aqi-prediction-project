"""Isolated experiment directories and reproducibility metadata."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import uuid

from paths import ALIGNED_CSV, GEE_DIR, ROOT


def add_run_args(parser):
    parser.add_argument("--run-name", help="Unique name inside results/<out>/runs/")
    parser.add_argument("--folds", nargs="+", help="Run selected fold names, e.g. loco:Mumbai")


def select_splits(splits, selected=None):
    splits = list(splits)
    if selected:
        unknown = set(selected) - {name for name, _, _ in splits}
        if unknown:
            raise ValueError(f"Unknown folds: {sorted(unknown)}")
        splits = [s for s in splits if s[0] in selected]
    return splits


def start_run(args, family, df, splits):
    out_name = getattr(args, "out", family)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", out_name):
        raise ValueError("--out must be a simple directory name")
    name = args.run_name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("--run-name must contain only letters, numbers, underscores or hyphens")
    out = ROOT / "results" / out_name / ("smoke" if args.quick else "runs") / name
    out.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "openpyxl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = {}
    for source in sorted((ROOT / "src").rglob("*.py")):
        sources[str(source.relative_to(ROOT))] = sha256(source.read_bytes()).hexdigest()
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                    family=family, arguments=vars(args), python=platform.python_version(), packages=versions,
                    dataset_sha256=sha256(ALIGNED_CSV.read_bytes()).hexdigest(), source_sha256=sources,
                    weather_sha256={city: sha256((GEE_DIR / f"{city}.csv").read_bytes()).hexdigest() for city in sorted(df.city.unique())},
                    rows=len(df), stations=int(df.station_id.nunique()), cities=sorted(df.city.unique()),
                    folds=[name for name, _, _ in splits],
                    validation="latest training dates, purged window boundary, fit-only normalization" if family != "baselines" else None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {out}", flush=True)
    return out


def complete_run(out):
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
