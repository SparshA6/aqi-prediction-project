"""Historical next-day forecast using observed AQI history, never future weather."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from baselines import make_model
from data import make_splits
from paths import ALIGNED_CSV, ROOT

FEATURES = ["latest_aqi", "lag1", "lag2", "lag7", "mean3", "doy_sin", "doy_cos", "lat", "lon"]


def forecast_table(aligned):
    """One row forecasts D+1 with AQI observed through the end of D.

    Consecutive calendar dates are looked up explicitly; missing days never
    become an accidental one-row lag. No environmental retrieval is required.
    """
    a = aligned.copy()
    a["date"] = pd.to_datetime(a.date)
    a = a[a.AQI_eod.notna() & ~a.aqi_flat_day].copy()
    if a.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate station/date rows")
    target = a.set_index(["station_id", "date"]).AQI_eod
    def lookup(days):
        index = pd.MultiIndex.from_arrays([a.station_id, a.date + pd.Timedelta(days=days)])
        return target.reindex(index).to_numpy()
    out = a[["date", "station_id", "city", "station_name", "station_lat", "station_lon"]].copy()
    out["feature_date"] = a.date
    out["date"] = a.date + pd.Timedelta(days=1)
    out["issue_time_utc"] = out.date
    out["latest_aqi"] = a.AQI_eod
    for lag in (1, 2, 7):
        out[f"lag{lag}"] = lookup(-lag)
    out["mean3"] = np.nanmean(np.column_stack([a.AQI_eod, lookup(-1), lookup(-2)]), axis=1)
    out["y"] = lookup(1)
    out["lat"], out["lon"] = a.station_lat, a.station_lon
    doy = out.date.dt.dayofyear
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["month"] = out.date.dt.month
    return out[out.y.notna()].sort_values(["date", "station_id"]).reset_index(drop=True)


def metrics(y, prediction):
    return dict(n=len(y), MAE=float(mean_absolute_error(y, prediction)),
                RMSE=float(np.sqrt(mean_squared_error(y, prediction))), R2=float(r2_score(y, prediction)),
                bias=float(np.mean(np.asarray(prediction) - y)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=["Ridge", "RandomForest", "XGBoost"], default=["Ridge", "RandomForest", "XGBoost"])
    ap.add_argument("--run-name")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    df = forecast_table(pd.read_csv(ALIGNED_CSV))
    _, tr, evs = next(make_splits(df, "chrono"))
    val, te = evs["val"], evs["test"]
    if min(len(tr), len(val), len(te)) < 30:
        raise SystemExit("Insufficient chronological training/calibration/test data")
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if Path(name).name != name or name in (".", ".."):
        ap.error("Invalid run name")
    out = ROOT / "results" / "forecast" / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    rows, fitted, predictions = [], {}, {}
    for model in args.models:
        estimator = make_model(model, args.jobs)
        estimator.fit(df.loc[tr, FEATURES], df.loc[tr, "y"])
        fitted[model] = estimator
        predictions[model] = {split: estimator.predict(df.loc[idx, FEATURES]) for split, idx in evs.items()}
        rows.append(dict(model=model, split="val", **metrics(df.loc[val, "y"].to_numpy(), predictions[model]["val"])))
    # Select using September only, calibrate using October, and keep test untouched.
    selection = val[df.loc[val, "date"].dt.month.to_numpy() == 9]
    calibration = val[df.loc[val, "date"].dt.month.to_numpy() == 10]
    if min(len(selection), len(calibration)) < 30:
        raise SystemExit("Insufficient separate model-selection/calibration rows")
    fitted["Persistence"] = None
    predictions["Persistence"] = {split: df.loc[idx, "latest_aqi"].to_numpy() for split, idx in evs.items()}
    choices = []
    for model in args.models + ["Persistence"]:
        select_pred = predictions[model]["val"][np.isin(val, selection)]
        choices.append((metrics(df.loc[selection, "y"].to_numpy(), select_pred)["RMSE"], model))
    selected = min(choices)[1]
    cal_pred = predictions[selected]["val"][np.isin(val, calibration)]
    residuals = np.abs(cal_pred - df.loc[calibration, "y"].to_numpy())
    n = len(residuals)
    quantile = min(1., np.ceil((n + 1) * .9) / n)
    radius = float(np.quantile(residuals, quantile, method="higher"))
    for model in args.models:
        rows.append(dict(model=model, split="test", **metrics(df.loc[te, "y"].to_numpy(), predictions[model]["test"])))
    for split, idx in evs.items():
        rows.append(dict(model="Persistence", split=split, **metrics(df.loc[idx, "y"].to_numpy(), df.loc[idx, "latest_aqi"].to_numpy())))
    result = df.loc[te, ["date", "feature_date", "issue_time_utc", "station_id", "station_name", "city", "y", "latest_aqi"]].copy()
    result["pred"] = predictions[selected]["test"]
    result["lower"] = np.clip(result.pred - radius, 0, 500)
    result["upper"] = np.clip(result.pred + radius, 0, 500)
    result["mode"] = "next_day"
    result.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    joblib.dump(dict(model=fitted[selected], features=FEATURES, interval_radius=radius, selected=selected), out / "forecast_model.joblib")
    manifest = dict(status="complete", task="historical next-day forecast", selected_model=selected,
                    selection="lowest September RMSE including persistence", calibration="October absolute residuals, nominal 90% split-conformal interval",
                    calibration_caveat="Temporal shift and station dependence break exchangeability; coverage is empirical, not guaranteed",
                    test_interval_coverage=float(((result.y >= result.lower) & (result.y <= result.upper)).mean()),
                    features=FEATURES, dataset_sha256=sha256(ALIGNED_CSV.read_bytes()).hexdigest(),
                    rows=dict(train=len(tr), validation=len(val), test=len(te)),
                    limitations=["Historical 2023 backtest, not live forecasts", "Assumes prior day's AQI is received by forecast issue time",
                                 "No reanalysis or future satellite retrieval enters forecast", "Single chronological test period"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Forecast saved: {out}")
    print(f"Selected on validation: {selected}; empirical test interval coverage {manifest['test_interval_coverage']:.1%}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
