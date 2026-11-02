"""Date-based inner validation with a purge for overlapping input windows."""

import numpy as np
import pandas as pd


def temporal_validation_split(dates, window=7, fraction=0.15):
    """Hold out the latest dates; purge fit rows whose windows touch validation.

    Returns boolean fit/validation masks in the caller's row order. This split
    uses dates only, never target values or an outer evaluation split.
    """
    if window < 1 or not 0 < fraction < 1:
        raise ValueError("window must be positive and fraction must be between 0 and 1")
    dates = pd.DatetimeIndex(dates).normalize()
    if dates.hasnans:
        raise ValueError("Training dates contain NaT")
    unique = dates.unique().sort_values()
    n_val = max(1, int(np.ceil(len(unique) * fraction)))
    if len(unique) <= n_val:
        raise ValueError("Not enough training dates for inner validation")
    cutoff = unique[-n_val]
    fit = np.asarray(dates < cutoff - pd.Timedelta(days=window - 1))
    val = np.asarray(dates >= cutoff)
    if fit.sum() < 2 or val.sum() < 2:
        raise ValueError("Not enough rows after purging overlapping windows")
    return fit, val
