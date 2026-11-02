"""Metrics, pooled and stratified. Long-format rows so results can be sliced any way afterwards."""

import numpy as np
import pandas as pd


def _m(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return dict(n=int(len(y)), MAE=np.nan, RMSE=np.nan, R2=np.nan, bias=np.nan)
    err = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                R2=float(1 - (err ** 2).sum() / ss) if ss > 0 else np.nan, bias=float(err.mean()))


SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    "sat_complete": lambda d: (d["satellite_bands_present"] == 3).values,     # all 3 satellite bands available that day
    "has_lag1": lambda d: d["aqi_l1"].notna().values,                          # rows persistence can be scored on
}


def evaluate(d, pred):
    """d: rows of the evaluated split (must have y, city, band, season); pred: array aligned with d.
    -> list of dict rows: subset x level(overall|city|band|season|macro_city|episode)"""
    d = d.reset_index(drop=True)
    y, p = d["y"].values, np.asarray(pred, dtype=float)
    rows = []
    for sname, fn in SUBSETS.items():
        mask = fn(d)
        if mask.sum() < 2:
            continue
        ds, ys, ps = d[mask], y[mask], p[mask]
        rows.append(dict(subset=sname, level="overall", group="all", **_m(ys, ps)))
        per_city = []
        for c, idx in ds.groupby("city").indices.items():
            r = _m(ys[idx], ps[idx])
            rows.append(dict(subset=sname, level="city", group=c, **r))
            if r["n"] >= 30:
                per_city.append(r)
        if per_city:
            rows.append(dict(subset=sname, level="macro_city", group="mean over cities",
                             n=int(sum(r["n"] for r in per_city)),
                             **{k: float(np.nanmean([r[k] for r in per_city])) for k in ("MAE", "RMSE", "R2", "bias")}))
        for col in ("band", "season"):
            for g, idx in ds.groupby(col).indices.items():
                rows.append(dict(subset=sname, level=col, group=g, **_m(ys[idx], ps[idx])))
        rows.append(dict(subset=sname, level="episode", group="AQI >= 300", **_m(ys[ys >= 300], ps[ys >= 300])))
        rows.append(dict(subset=sname, level="episode", group="AQI >= 400", **_m(ys[ys >= 400], ps[ys >= 400])))
    return rows
