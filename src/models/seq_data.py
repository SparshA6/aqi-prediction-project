"""
Sliding-window tensors for the sequence models (environment-only).

For each (station, day) row of the baseline table we build the window of the previous W-1 days plus the
day itself, taken from that station's grid cell: per-day features = satellite (NO2, HCHO, AOD; NaN when
cloud-masked) + same-day weather (+ optional seasonal sin/cos of that day). No AQI history is used anywhere
in the input. Days before 2023-01-01 do not exist -> NaN (treated as missing).

Row order == load_dataset() order, so indices line up with the baseline splits and predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from data import SAT, SEASONAL, WX, YEAR_DAYS, _weather_frame
from paths import GEE_DIR

MAX_WINDOW = 7
STEP_FEATURES = SAT + WX + SEASONAL            # columns of the raw per-day array
N_SAT = len(SAT)


def build_windows(df, max_window=MAX_WINDOW):
    """-> array (n_rows, max_window, len(STEP_FEATURES)); step -1 is the day itself, step 0 the oldest day."""
    X = np.full((len(df), max_window, len(STEP_FEATURES)), np.nan, dtype=np.float32)
    pos_of = df.index.values
    for city, sub in df.groupby("city"):
        g = pd.read_csv(GEE_DIR / f"{city}.csv", parse_dates=["date"])
        for sid, s in sub.groupby("station_id"):
            gl, go = s.iloc[0][["grid_lat", "grid_lon"]]
            cell = g[np.isclose(g.latitude, gl) & np.isclose(g.longitude, go)]
            w = _weather_frame(cell)[STEP_FEATURES].to_numpy(dtype=np.float32)          # (365, F)
            padded = np.vstack([np.full((max_window - 1, w.shape[1]), np.nan, np.float32), w])
            doy0 = YEAR_DAYS.get_indexer(s.date)                                       # 0-based day index in the year
            for k in range(max_window):                                                   # step k = day (d - (max_window-1-k))
                X[s.index.values, k, :] = padded[doy0 + k, :]
    return X


class Preprocessor:
    """Fit on training rows only: standardise each feature over the training windows, NaN -> 0 after scaling,
    plus one missing-indicator per satellite band per step."""

    def __init__(self, use_season):
        self.cols = list(range(len(SAT) + len(WX))) + (list(range(len(SAT) + len(WX), len(STEP_FEATURES))) if use_season else [])

    def fit(self, X_train):
        Z = X_train[:, :, self.cols].reshape(-1, len(self.cols))
        self.mu = np.nanmean(Z, axis=0)
        self.sd = np.nanstd(Z, axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X, window):
        Xw = X[:, -window:, :][:, :, self.cols]
        miss = np.isnan(Xw[:, :, :N_SAT]).astype(np.float32)
        Z = (Xw - self.mu) / self.sd
        Z = np.nan_to_num(Z, nan=0.0)
        return np.concatenate([Z, miss], axis=2).astype(np.float32)


if __name__ == "__main__":
    from data import load_dataset
    df = load_dataset()
    X = build_windows(df)
    print(X.shape, "NaN share per step feature (last step):")
    print(pd.Series(np.isnan(X[:, -1, :]).mean(0), index=STEP_FEATURES).round(3).to_string())
    # spot check: last step equals the baseline table's same-day columns, step -2 equals the *_l1 lags
    for j, c in enumerate(STEP_FEATURES):
        assert np.allclose(X[:, -1, j], df[c].values.astype(np.float32), equal_nan=True, atol=1e-5), c
    for c in ("temp_c", "rh", "wspd", "pressure_kpa", "precip_mm"):
        j = STEP_FEATURES.index(c)
        assert np.allclose(X[:, -2, j], df[f"{c}_l1"].values.astype(np.float32), equal_nan=True, atol=1e-5), c
        assert np.allclose(X[:, -3, j], df[f"{c}_l2"].values.astype(np.float32), equal_nan=True, atol=1e-5), c
    print("window alignment check vs baseline table: OK")
