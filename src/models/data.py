"""
Feature table + evaluation splits for the baseline models.

One row per (station, day). Target = AQI_eod (the CPCB hourly AQI at 05:00 IST next morning, whose
trailing 24 h window matches the UTC weather day -- see docs/PROJECT_BRIEF.md 6g). Stuck-sensor days
(aqi_flat_day) and days without an AQI_eod reading are dropped.

Feature groups (the ablation ladder from the project brief):
    SAT      satellite columns (NO2, HCHO, AOD) -- NaN when cloud-masked
    WX       same-day weather at the station's grid cell
    SPATIAL  station latitude / longitude
    TEMPORAL day-of-year (sin/cos) + previous days' weather (lags 1-2, 3-day wind/rain)
    HIST     the station's own previous AQI values (yesterday, 2 days ago, 3-day mean, a week ago)
HIST is kept OUT of the main ladder: it is a forecasting input that would dominate any explanation
of the environmental drivers, so it is reported as a separate extra step.

Every lag is a shift within one station's complete daily series, so a row can only see the past.
selfcheck() proves this on random rows against the raw files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from paths import ALIGNED_CSV, GEE_DIR

TARGET = "AQI_eod"
def calendar_index():
    dates = pd.to_datetime(pd.read_csv(ALIGNED_CSV, usecols=["date"])["date"])
    return pd.date_range(f"{dates.min().year}-01-01", f"{dates.max().year}-12-31", freq="D")

YEAR_DAYS = calendar_index()

SAT = ["NO2", "HCHO", "AOD"]
WX = ["temp_c", "rh", "wspd", "wdir_sin", "wdir_cos", "pressure_kpa", "precip_mm"]
SPATIAL = ["lat", "lon"]
SEASONAL = ["doy_sin", "doy_cos"]
WX_LAGS = [f"{v}_l{k}" for v in ("temp_c", "rh", "wspd", "precip_mm", "pressure_kpa") for k in (1, 2)] + ["wspd_3d", "precip_3d"]
TEMPORAL = SEASONAL + WX_LAGS
HIST = ["aqi_l1", "aqi_l2", "aqi_l3mean", "aqi_l7"]

FEATURE_SETS = {
    "S": SAT,
    "W": WX,
    "SW": SAT + WX,
    "SW+G": SAT + WX + SPATIAL,
    "SW+G+T": SAT + WX + SPATIAL + TEMPORAL,
    "SW+G+T+H": SAT + WX + SPATIAL + TEMPORAL + HIST,
}

BANDS = [(0, 50, "1 Good"), (51, 100, "2 Satisfactory"), (101, 200, "3 Moderate"),
         (201, 300, "4 Poor"), (301, 400, "5 Very Poor"), (401, 10_000, "6 Severe")]


def band(v):
    for lo, hi, name in BANDS:
        if 0 <= v <= hi:
            return name
    return np.nan


def season(month):
    return {12: "winter", 1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon",
            6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon"}[month]


def _weather_frame(g):
    """raw gee rows of ONE grid cell (sorted by date) -> derived weather + lags on a complete daily index."""
    g = g.set_index("date").sort_index().reindex(YEAR_DAYS)
    t = g.temperature_2m - 273.15
    td = g.dewpoint_temperature_2m - 273.15
    w = pd.DataFrame(index=YEAR_DAYS)
    w["NO2"], w["HCHO"], w["AOD"] = g.NO2, g.HCHO, g.AOD
    w["temp_c"] = t
    w["rh"] = (100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))).clip(0, 100)
    w["wspd"] = np.hypot(g.u_component_of_wind_10m, g.v_component_of_wind_10m)
    wdir = (270 - np.degrees(np.arctan2(g.v_component_of_wind_10m, g.u_component_of_wind_10m))) % 360   # direction wind blows FROM
    w["wdir_sin"], w["wdir_cos"] = np.sin(np.radians(wdir)), np.cos(np.radians(wdir))
    w["pressure_kpa"] = g.surface_pressure / 1000
    w["precip_mm"] = (g.total_precipitation_sum * 1000).clip(lower=0)      # ERA5-Land has tiny negative noise
    for v in ("temp_c", "rh", "wspd", "precip_mm", "pressure_kpa"):
        for k in (1, 2):
            w[f"{v}_l{k}"] = w[v].shift(k)
    w["wspd_3d"] = w["wspd"].shift(1).rolling(3, min_periods=2).mean()     # wind of the 3 previous days
    w["precip_3d"] = w["precip_mm"].shift(1).rolling(3, min_periods=2).sum()
    doy = w.index.dayofyear.values
    w["doy_sin"], w["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return w


def load_dataset():
    al = pd.read_csv(ALIGNED_CSV, parse_dates=["date"])
    al = al[al[TARGET].notna() & ~al["aqi_flat_day"]].copy()

    frames = []
    for city, sub in al.groupby("city"):
        g = pd.read_csv(GEE_DIR / f"{city}.csv", parse_dates=["date"])
        for sid, s in sub.groupby("station_id"):
            gl, go = s.iloc[0][["grid_lat", "grid_lon"]]
            cell = g[np.isclose(g.latitude, gl) & np.isclose(g.longitude, go)]
            w = _weather_frame(cell)
            hist = s.set_index("date")[TARGET].reindex(YEAR_DAYS)
            w["aqi_l1"], w["aqi_l2"], w["aqi_l7"] = hist.shift(1), hist.shift(2), hist.shift(7)
            w["aqi_l3mean"] = pd.concat([hist.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)   # NaN only if all 3 missing
            w = w.loc[s.date]
            w.index = s.index
            frames.append(w)
    feat = pd.concat(frames)

    df = al.join(feat[[c for c in feat.columns if c not in al.columns]])
    df["lat"], df["lon"] = df["station_lat"], df["station_lon"]
    df["y"] = df[TARGET]
    df["month"] = df.date.dt.month
    df["season"] = df.month.map(season)
    df["band"] = df.y.map(band)
    return df.sort_values(["date", "station_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ splits
def make_splits(df, protocol):
    """yield (fold_name, train_idx, {eval_name: idx})"""
    d = df.date
    year = d.max().year
    val_start, test_start = f"{year}-09-01", f"{year}-11-01"
    if protocol == "chrono":
        tr = df.index[d < val_start]
        yield "chrono", tr, {"val": df.index[(d >= val_start) & (d < test_start)], "test": df.index[d >= test_start]}
    elif protocol == "loco":
        for city in sorted(df.city.unique()):
            yield f"loco:{city}", df.index[df.city != city], {"test": df.index[df.city == city]}
    elif protocol == "season_cv":
        # 4 folds of interleaved months; every fold's test months are 3 months apart, so each fold
        # trains on all seasons and tests on 3 separate months
        for k in range(4):
            months = [k + 1, k + 5, k + 9]
            te = df.index[df.month.isin(months)]
            yield f"months{months}", df.index.difference(te), {"test": te}
    else:
        raise ValueError(protocol)


# ------------------------------------------------------------------ leakage / correctness self-check
def selfcheck(df, n=300, seed=0):
    rng = np.random.default_rng(seed)
    aligned = pd.read_csv(ALIGNED_CSV, parse_dates=["date"]).set_index(["station_id", "date"])
    bad = 0
    cache = {}
    for i in rng.choice(len(df), size=min(n, len(df)), replace=False):
        r = df.iloc[i]
        # 1) AQI history must be yesterday's / last week's target of the SAME station (or NaN if none)
        for col, k in (("aqi_l1", 1), ("aqi_l2", 2), ("aqi_l7", 7)):
            key = (r.station_id, r.date - pd.Timedelta(days=k))
            exp = aligned[TARGET].get(key, np.nan)
            if key in aligned.index and aligned.loc[key, "aqi_flat_day"]:
                exp = np.nan if False else exp
            got = r[col]
            ok = (np.isnan(got) and (np.isnan(exp) or (key in aligned.index and aligned.loc[key, "aqi_flat_day"]))) or np.isclose(got, exp)
            bad += not ok
        # 2) lagged weather must be the raw cell value one day earlier
        if r.city not in cache:
            cache[r.city] = pd.read_csv(GEE_DIR / f"{r.city}.csv", parse_dates=["date"])
        g = cache[r.city]
        gd = g[np.isclose(g.latitude, r.grid_lat) & np.isclose(g.longitude, r.grid_lon)].set_index("date")
        prev = gd.loc[r.date - pd.Timedelta(days=1)] if (r.date - pd.Timedelta(days=1)) in gd.index else None
        if prev is not None and not np.isnan(r.wspd_l1):
            bad += not np.isclose(r.wspd_l1, np.hypot(prev.u_component_of_wind_10m, prev.v_component_of_wind_10m))
            bad += not np.isclose(r.temp_c_l1, prev.temperature_2m - 273.15)
        # 3) same-day weather equals the raw cell value that day
        cur = gd.loc[r.date]
        bad += not np.isclose(r.pressure_kpa, cur.surface_pressure / 1000)
    return bad


if __name__ == "__main__":
    df = load_dataset()
    print("rows", len(df), "| stations", df.station_id.nunique(), "| cities", df.city.nunique())
    print("NaN share per feature (only nonzero):")
    allf = sorted({c for f in FEATURE_SETS.values() for c in f})
    nn = df[allf].isna().mean()
    print(nn[nn > 0].round(3).to_string())
    print("leakage/correctness self-check violations (expect 0):", selfcheck(df))
    for p in ("chrono", "loco", "season_cv"):
        print(p, [(name, len(tr), {k: len(v) for k, v in ev.items()}) for name, tr, ev in make_splits(df, p)][:3], "...")
