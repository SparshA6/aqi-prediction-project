"""Rebuild share/: the local demo ZIP and the public static page.

This publishes nothing. To update the public demo, copy share/website into the
demo repository and push it.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).with_name("demo_package")
SHARE = ROOT / "share"
WEBSITE = SHARE / "website"
PREFIX = "AQI_Local_Demo"
PDFS = ["AQI_Methodology_Submission.pdf", "AQI_Results_and_Discussion_Submission.pdf"]
ALIGNED = "data/processed/multiyear_2023_2024_2025/aligned_dataset.csv"
ALIGNED_COLUMNS = ["date", "station_id", "AQI_eod", "aqi_flat_day"]
AUDIT_FIELDS = ["kind", "model", "rows", "validation"]

FORECAST_2025 = "results/forecast_2025/runs/locked-2025-v1"
FORECAST_2026 = "results/forecast_2026/runs/locked-2026-v1"
HOLDOUT = "results/year_holdout/runs/2023-to-2024-v1"
FROZEN_2025 = "results/forecast_development/runs/2025-ready-v1"
FROZEN_2026 = "results/forecast_development/runs/2026-ready-v1"
FOLLOWTHROUGH = "results/followthrough/oct12-v1"

# Project files copied into the ZIP unchanged, at the same relative path.
PROJECT_FILES = [
    "config/stations.csv",
    "data/processed/forecast_2026/coverage.csv",
    *[f"docs/{name}" for name in PDFS],
    *[f"{FORECAST_2025}/{name}" for name in ["group_metrics.csv", "manifest.json", "metrics.csv", "predictions.csv.gz", "verification.json"]],
    *[f"{FORECAST_2026}/{name}" for name in ["city_metrics.csv", "manifest.json", "metrics.csv", "predictions.csv.gz", "spike_metrics.csv", "verification.json"]],
    *[f"{FROZEN_2025}/{name}" for name in ["forecast_model.joblib", "manifest.json", "models/GRU-28days-s0.pt", "models/GRU-28days-s1.pt", "models/GRU-28days-s2.pt"]],
    *[f"{FROZEN_2026}/{name}" for name in ["development_metrics.csv", "manifest.json"]],
    *[f"{HOLDOUT}/{name}" for name in ["manifest.json", "metrics.csv", "predictions.csv.gz", "verification.json"]],
    "results/attention_stability/gains.csv",
    "results/prospective/status.json",
    f"{FOLLOWTHROUGH}/summary.json",
    f"{FOLLOWTHROUGH}/Results_Report.txt",
    "src/dashboard/app.py",
    "src/dashboard/index.html",
    *[f"src/models/{name}" for name in ["evaluate_forecast_2025.py", "forecast_development.py", "seq_models.py", "validation.py", "year_holdout.py"]],
]
# demo_package files placed at the top of the ZIP.
PACKAGE_FILES = {"README.md": "LOCAL_README.md", "requirements.txt": "requirements.txt", "run_unix.sh": "run_unix.sh", "run_windows.bat": "run_windows.bat"}
TEXT_SUFFIXES = {".csv", ".html", ".json", ".md", ".py", ".txt"}


def audit_manifests():
    """Dashboard fields of each faithfulness manifest; the full files hold local paths."""
    manifests = {}
    for correlation in sorted((ROOT / "results/faithfulness").glob("*/correlations.csv")):
        path = correlation.with_name("manifest.json")
        manifest = json.loads(path.read_text())
        manifests[path.relative_to(ROOT).as_posix()] = json.dumps({k: manifest[k] for k in AUDIT_FIELDS if k in manifest}, indent=2)
    return manifests


def local_paths(data):
    """Paths from this computer found in the bytes of a file about to be shared."""
    found = []
    for path in [str(ROOT), ROOT.as_posix(), str(Path.home()), Path.home().as_posix()]:
        for form in {path, path.replace("\\", "\\\\")}:
            if form.encode() in data:
                found.append(form)
    return found


def checked(name, data):
    if Path(name).suffix in TEXT_SUFFIXES and local_paths(data):
        raise ValueError(f"{name} contains a path from this computer")
    return data


def build_zip(target):
    aligned = pd.read_csv(ROOT / ALIGNED, usecols=ALIGNED_COLUMNS)[ALIGNED_COLUMNS]
    members = {name: (ROOT / name).read_bytes() for name in PROJECT_FILES}
    members.update({name: (PACKAGE / source).read_bytes() for name, source in PACKAGE_FILES.items()})
    members.update({name: text.encode() for name, text in audit_manifests().items()})
    for correlation in sorted((ROOT / "results/faithfulness").glob("*/correlations.csv")):
        members[correlation.relative_to(ROOT).as_posix()] = correlation.read_bytes()
    members[ALIGNED] = aligned.to_csv(index=False, lineterminator="\r\n").encode()
    temporary = target.with_name(target.name + ".new")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            archive.writestr(f"{PREFIX}/{name}", checked(name, members[name]))
    temporary.replace(target)
    return len(members)


def main():
    WEBSITE.mkdir(parents=True, exist_ok=True)
    count = build_zip(SHARE / f"{PREFIX}.zip")
    page = WEBSITE / "index.html"
    subprocess.run([sys.executable, str(ROOT / "src/dashboard/app.py"), "--export", str(page), "--public-files"], check=True, cwd=ROOT)
    checked(page.name, page.read_bytes())
    shutil.copy2(SHARE / f"{PREFIX}.zip", WEBSITE / f"{PREFIX}.zip")
    shutil.copy2(PACKAGE / "PUBLIC_README.md", WEBSITE / "README.md")
    for name in ["LOCAL_README.md", "PUBLIC_README.md"]:
        shutil.copy2(PACKAGE / name, SHARE / name)
    for name in PDFS:
        shutil.copy2(ROOT / "docs" / name, WEBSITE / name)
    print(f"ZIP: {count} files, {(SHARE / f'{PREFIX}.zip').stat().st_size / 1e6:.1f} MB")
    print(f"Page: {page.stat().st_size / 1e6:.1f} MB")
    print(f"Built in {SHARE}. Nothing was published.")


if __name__ == "__main__":
    main()
