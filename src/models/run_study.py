"""Run the full corrected experiment grid sequentially, with resumable status."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paths import ALIGNED_CSV, GEE_DIR


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="corrected-v2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum() or args.seeds < 1:
        ap.error("Use a simple study name and positive seed count")
    directory = ROOT / "results" / "studies" / args.name
    status_path = directory / "status.json"
    if args.resume:
        status = json.loads(status_path.read_text())
        if args.device != status.get("training_device", "cpu"):
            raise SystemExit("Device changed; create a new study to keep execution provenance explicit")
        if args.seeds != status["seeds"]:
            raise SystemExit("Seed count changed; create a new study")
        for relative, digest in status.get("source_sha256", {}).items():
            if Path(relative).name in {"run_study.py", "audit_attention.py", "audit_groups.py", "check_results.py"} or Path(relative).name.startswith("summarize_"):
                continue
            if sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise SystemExit("Model source changed; create a new study instead of mixing results")
        if status["dataset_sha256"] != sha256(ALIGNED_CSV.read_bytes()).hexdigest():
            raise SystemExit("Dataset changed; create a new study instead of mixing runs")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        jobs = []
        for protocol in ("chrono", "loco", "season_cv"):
            for family in ("baselines", "attention", "sequence", "spatial"):
                name = f"{args.name}-{family}-{protocol}"
                script = "run_sequence.py" if family == "attention" else f"run_{family}.py"
                command = [sys.executable, "-B", str(ROOT / "src" / "models" / script), "--protocol", protocol, "--run-name", name]
                if family == "baselines":
                    command += ["--jobs", "2"]
                else:
                    command += ["--seeds", str(args.seeds), "--threads", "2", "--save-models", "--device", args.device]
                if family == "attention":
                    command += ["--out", "attention", "--archs", "Uniform", "TA", "FA", "TA+FA", "--sets", "SW-w7", "SWT-w7"]
                jobs.append(dict(family=family, protocol=protocol, name=name, command=command, status="queued"))
        status = dict(study=args.name, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                      dataset_sha256=sha256(ALIGNED_CSV.read_bytes()).hexdigest(), seeds=args.seeds, training_device=args.device, jobs=jobs,
                      source_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src/models").glob("*.py"))})
        with zipfile.ZipFile(directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in (ROOT / "src").rglob("*.py"):
                archive.write(p, str(p.relative_to(ROOT)))
        shutil.copy2(ALIGNED_CSV, directory / "aligned_snapshot.csv")
        save(status_path, status)
    status["status"] = "running"
    save(status_path, status)
    for job in status["jobs"]:
        if job["status"] != "complete":
            job["command"][0] = sys.executable
            if job["family"] != "baselines":
                if "--device" in job["command"]:
                    job["command"][job["command"].index("--device") + 1] = args.device
                else:
                    job["command"] += ["--device", args.device]
        if job["status"] == "complete":
            continue
        if job["status"] != "queued":
            # A interrupted run may already have generated artifacts. Preserve them.
            job["name"] += "-retry-" + datetime.now(timezone.utc).strftime("%H%M%S")
            index = job["command"].index("--run-name") + 1
            job["command"][index] = job["name"]
        job.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
        save(status_path, status)
        print(f"Starting {job['family']} / {job['protocol']}", flush=True)
        with (directory / f"{job['name']}.log").open("w", encoding="utf-8") as log:
            code = subprocess.call(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_dir = ROOT / "results" / job["family"] / "runs" / job["name"]
        if code == 0:
            with (directory / f"{job['name']}-check.log").open("w", encoding="utf-8") as log:
                code = subprocess.call([sys.executable, "-B", str(ROOT / "src/models/check_results.py"), "--results-dir", str(result_dir)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        job.update(status="complete" if code == 0 else "failed", exit_code=code,
                   results_dir=str(result_dir), ended_utc=datetime.now(timezone.utc).isoformat())
        save(status_path, status)
        # Audit each new complete chronological attention/spatial experiment automatically.
        if code == 0 and job["protocol"] == "chrono" and job["family"] in ("attention", "spatial"):
            audits = [["audit_attention.py", "--models-dir", str(result_dir / "models"), "--sample", "1000", "--random-repeats", "20"]] if job["family"] == "attention" else []
            audits.append(["audit_groups.py", "--kind", "feature" if job["family"] == "attention" else "spatial", "--model", "FA" if job["family"] == "attention" else "S-SA", "--models-dir", str(result_dir / "models"), "--sample", "300", "--shap-samples", "128"])
            job["audit_exit_codes"] = []
            for i, audit in enumerate(audits):
                with (directory / f"{job['name']}-audit-{i}.log").open("w", encoding="utf-8") as log:
                    job["audit_exit_codes"].append(subprocess.call([sys.executable, "-B", str(ROOT / "src/models" / audit[0]), *audit[1:]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            save(status_path, status)
        if code != 0:
            status["status"] = "needs_attention"
            save(status_path, status)
            raise SystemExit(f"Experiment failed; see {directory}. Resume preserves existing results.")
    status.update(status="complete", ended_utc=datetime.now(timezone.utc).isoformat())
    if any(any(code != 0 for code in j.get("audit_exit_codes", [])) for j in status["jobs"]):
        status["status"] = "experiments_complete_audit_needs_attention"
    save(status_path, status)
    print(f"Study completed: {directory}")


if __name__ == "__main__":
    main()
