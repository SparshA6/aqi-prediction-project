"""
Neighbouring-cell patches for the spatiotemporal attention model (environment-only).

For each (station, day) row we take a (2r+1) x (2r+1) patch of grid cells around the station's matched grid cell,
spaced `stride` grid steps apart (grid step = 0.03 deg; ERA5-Land's native resolution is ~0.1 deg, so stride 3 makes
neighbouring cells carry genuinely different weather -- with stride 1 they are mostly identical copies).
Every cell contributes its own 7-day window of satellite + weather (same features as seq_data). Cells that fall
outside the city's extraction grid (sea, bbox edge) are masked. No absolute lat/lon is given to the model -- only each
cell's offset from the station (scaled to [-1, 1]) -- so the model can attend to *where* a cell is without being
able to memorise which station it is.

Cell order: row-major over (dy, dx) offsets; the centre cell is index (C-1)//2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from data import SAT, SEASONAL, WX, YEAR_DAYS, _weather_frame
from paths import GEE_DIR
from seq_data import MAX_WINDOW, N_SAT, STEP_FEATURES

GRID_STEP = 0.03


def patch_offsets(radius, stride):
    return [(i * stride, j * stride) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]


def build_patches(df, radius=1, stride=3, max_window=MAX_WINDOW):
    """-> P (n, C, T, F) raw features (NaN where missing/absent), present (n, C) bool, offsets (C, 2) in [-1, 1]"""
    offs = patch_offsets(radius, stride)
    C = len(offs)
    P = np.full((len(df), C, max_window, len(STEP_FEATURES)), np.nan, dtype=np.float32)
    present = np.zeros((len(df), C), dtype=bool)
    for city, sub in df.groupby("city"):
        g = pd.read_csv(GEE_DIR / f"{city}.csv", parse_dates=["date"])
        lat0, lon0 = g.latitude.min(), g.longitude.min()
        g["iy"] = np.round((g.latitude - lat0) / GRID_STEP).astype(int)
        g["ix"] = np.round((g.longitude - lon0) / GRID_STEP).astype(int)
        cells = {k: v for k, v in g.groupby(["iy", "ix"])}
        cache = {}

        def cell_array(key):
            if key not in cache:
                if key not in cells:
                    cache[key] = None
                else:
                    w = _weather_frame(cells[key])[STEP_FEATURES].to_numpy(dtype=np.float32)
                    cache[key] = np.vstack([np.full((max_window - 1, w.shape[1]), np.nan, np.float32), w])
            return cache[key]

        for sid, s in sub.groupby("station_id"):
            gl, go = s.iloc[0][["grid_lat", "grid_lon"]]
            cy, cx = int(round((gl - lat0) / GRID_STEP)), int(round((go - lon0) / GRID_STEP))
            doy0 = YEAR_DAYS.get_indexer(s.date)
            rows = s.index.values
            for k, (dy, dx) in enumerate(offs):
                arr = cell_array((cy + dy, cx + dx))
                if arr is None:
                    continue
                present[rows, k] = True
                for t in range(max_window):
                    P[rows, k, t, :] = arr[doy0 + t, :]
    off_arr = np.asarray(offs, dtype=np.float32) / max(1, radius * stride)
    return P, present, off_arr


class SpatialPreprocessor:
    """Standardise over the training rows' PRESENT cells (per feature); NaN -> 0; satellite missing indicators only for
    present cells; append offset (dy, dx) channels and a final `present` channel (used by the network as the mask)."""

    def __init__(self, use_season):
        n_env = len(SAT) + len(WX)
        self.cols = list(range(n_env)) + (list(range(n_env, len(STEP_FEATURES))) if use_season else [])

    def fit(self, P_train, present_train):
        Z = P_train[present_train][:, :, self.cols].reshape(-1, len(self.cols))      # (n_present_cells * T, F)
        self.mu, self.sd = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, P, present, offsets, window=MAX_WINDOW):
        Pw = P[:, :, -window:, :][:, :, :, self.cols]
        pres = present[:, :, None, None]
        miss = (np.isnan(Pw[..., :N_SAT]) & pres).astype(np.float32)
        Z = np.nan_to_num((Pw - self.mu) / self.sd, nan=0.0)
        n, C, T, _ = Z.shape
        off = np.broadcast_to(offsets[None, :, None, :], (n, C, T, 2))
        pr = np.broadcast_to(present[:, :, None, None].astype(np.float32), (n, C, T, 1))
        return np.concatenate([Z, miss, off, pr], axis=3).astype(np.float32)


if __name__ == "__main__":
    from data import load_dataset
    from seq_data import build_windows
    df = load_dataset()
    for radius, stride in ((1, 3), (2, 3)):
        P, present, offs = build_patches(df, radius, stride)
        C = P.shape[1]
        c = (C - 1) // 2
        print(f"radius {radius} stride {stride}: P {P.shape}  cells present per row: mean {present.sum(1).mean():.1f}/{C}, "
              f"rows with a full patch {present.all(1).mean():.1%}, centre always present {present[:, c].all()}")
        by_city = pd.Series(present.sum(1), index=df.index).groupby(df.city).mean().round(1)
        print("  mean cells present by city:", by_city.to_dict())
        if radius == 1:
            X = build_windows(df)
            assert np.allclose(P[:, c], X, equal_nan=True, atol=1e-5), "centre cell != station-cell windows"
            print("  centre cell equals the single-cell windows used so far: OK")
        # does the neighbourhood carry information the centre cell lacks? relative spread of same-day values across present cells
        for name in ("temp_c", "wspd", "NO2", "AOD"):
            j = STEP_FEATURES.index(name)
            v = np.where(present, P[:, :, -1, j], np.nan)
            ctr = P[:, c, -1, j]
            sd_within = np.nanstd(v, axis=1)
            print(f"  {name:8s} within-patch sd (same day) mean {np.nanmean(sd_within):.4g} | across-row sd of centre {np.nanstd(ctr):.4g} | "
                  f"ratio {np.nanmean(sd_within) / np.nanstd(ctr):.3f}")
