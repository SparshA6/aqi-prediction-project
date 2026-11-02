"""
Join the Earth Engine satellite/meteorology grids (from gee_extract.py) with
CPCB station AQI data (from cpcb_downloader.py) into one modeling-ready
aligned dataset, matching each station to its nearest satellite/met grid
point by coordinates and merging on date.

Inputs:
    data/raw/gee/<City>.csv          -- one per city, from gee_extract.py
    config/stations.csv              -- station_id, station_name, city, climatic_zone,
                                        lat, lon, data_dir, notes
    <data_dir>/*.xlsx                -- CPCB monthly exports, from cpcb_downloader.py

Output:
    data/processed/aligned_dataset.csv -- one row per (station, date):
        date, city, climatic_zone, station_id, station_name,
        station_lat, station_lon, grid_lat, grid_lon, match_distance_km,
        AQI_eod, AQI_mean, AQI_max, AQI_min, AQI_valid_hours, AQI_band, aqi_flat_day,
        satellite_bands_present,
        <all satellite/meteorology feature columns from the matched grid point>

Usage:
    python src/combine_with_aqi.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from paths import ALIGNED_CSV, GEE_DIR, ROOT, STATIONS_CSV

# ============================================================
# CONFIGURATION
# ============================================================

GEE_DATA_DIR = GEE_DIR
OUTPUT_PATH = ALIGNED_CSV

MAX_MATCH_DISTANCE_KM = 15  # stations farther than this from any grid point are dropped, with a warning
DAILY_AGGREGATION = "eod"  # AQI_band uses the same target as the models
MIN_VALID_HOURS = 12  # a day needs at least this many non-null hourly readings to be kept

# CPCB reports hourly AQI in India Standard Time (UTC+5:30, no DST). The
# satellite/meteorology side (gee_extract.py's ERA5-Land / Sentinel-5P daily
# composites) is binned on UTC calendar days -- confirmed against Google's
# Earth Engine catalog docs for ECMWF/ERA5_LAND/DAILY_AGGR, which aggregate
# strictly 00:00-24:00 UTC regardless of local time. Left unaligned, a naive
# "same calendar date" join is off by up to 5.5 hours (e.g. IST 00:00-05:00
# on day D actually falls in UTC day D-1). ALIGN_TO_UTC_CALENDAR_DAY fixes
# this by shifting each hourly CPCB reading to UTC before aggregating, so
# both sides bin on the same UTC day. Set False to instead bin on the raw
# IST calendar day (i.e. reproduce the old, simpler, timezone-naive join).
ALIGN_TO_UTC_CALENDAR_DAY = True
# IST clock hour of the reading used for AQI_eod (see daily_aggregate): 05:00 IST closes the UTC day.
EOD_HOUR_IST = 5 if ALIGN_TO_UTC_CALENDAR_DAY else 23
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

EARTH_RADIUS_KM = 6371.0

# Satellite columns in the gee_extract.py output. Cloud-masked values are NaN there (the weather
# columns are always present), so a station-day can have 0-3 of these; see satellite_bands_present.
SATELLITE_COLUMNS = ["NO2", "HCHO", "AOD", "CO", "SO2", "O3"]

# India / CPCB official AQI category breakpoints
AQI_BANDS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 10_000, "Severe"),
]

MONTH_NAMES = (
    "January February March April May June July August September October "
    "November December"
).split()

# CPCB has used at least two filename conventions for the same underlying
# per-station-per-month export:
#   old:  aqi_hourly_station_level_<station>_<year>_<Month>_delhi_<year>.xlsx
#   new:  AQI_Data_Station_Level_Site_<id>_<Station_Name>_<Month>_<year>_Hourly.xlsx
# Both put a 4-digit year adjacent to a month name, just in opposite order,
# so match either order rather than depending on the rest of the filename.
_MONTHS_PATTERN = "|".join(MONTH_NAMES)
FILENAME_RE = re.compile(
    r"(?:(?P<year_first>\d{4})_(?P<month_a>"
    + _MONTHS_PATTERN
    + r")|(?P<month_b>"
    + _MONTHS_PATTERN
    + r")_(?P<year_second>\d{4}))",
    re.IGNORECASE,
)


# ============================================================
# CPCB PARSING
# ============================================================

def parse_cpcb_month_file(path):
    """One aqi_hourly_station_level_*.xlsx -> long DataFrame [date, hour_ist, aqi].

    `date` is the UTC calendar date (see ALIGN_TO_UTC_CALENDAR_DAY above) --
    the key this gets merged on against the satellite/met grid.
    """
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Can't parse year/month from filename: {path.name}")
    year_str = m.group("year_first") or m.group("year_second")
    month_str = m.group("month_a") or m.group("month_b")
    year = int(year_str)
    month = MONTH_NAMES.index(month_str.title()) + 1

    df = pd.read_excel(path, header=0)
    df = df.rename(columns={df.columns[0]: "day"})
    df = df.dropna(subset=["day"])
    df["day"] = df["day"].astype(int)

    hour_cols = [c for c in df.columns if c != "day"]
    long_df = df.melt(id_vars="day", value_vars=hour_cols, var_name="hour_str", value_name="aqi")
    long_df["hour_ist"] = long_df["hour_str"].astype(str).str.slice(0, 2).astype(int)

    timestamp_ist = pd.to_datetime(
        dict(year=year, month=month, day=long_df["day"], hour=long_df["hour_ist"]),
        errors="coerce",
    )
    long_df = long_df.assign(timestamp_ist=timestamp_ist).dropna(subset=["timestamp_ist"])

    if ALIGN_TO_UTC_CALENDAR_DAY:
        long_df["date"] = (long_df["timestamp_ist"] - IST_OFFSET).dt.normalize()
    else:
        long_df["date"] = long_df["timestamp_ist"].dt.normalize()

    return long_df[["date", "hour_ist", "aqi"]]


def load_station_hourly(station_row):
    data_dir = ROOT / station_row["data_dir"]
    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        print(f"  WARNING: no CPCB files found in {data_dir} for {station_row['station_id']}")
        return pd.DataFrame(columns=["date", "hour_ist", "aqi"])

    parts = []
    for f in files:
        try:
            parts.append(parse_cpcb_month_file(f))
        except ValueError as e:
            print(f"  WARNING: {e}")
    if not parts:
        return pd.DataFrame(columns=["date", "hour_ist", "aqi"])

    combined = pd.concat(parts, ignore_index=True)

    # CPCB has changed its export filename convention over time (old
    # "aqi_hourly_station_level_..." vs new "AQI_Data_Station_Level_..."),
    # so a folder can end up with two files covering the same month if it
    # was downloaded more than once. Collapse duplicate (date, hour) rows
    # rather than double-counting them into AQI_valid_hours.
    dupe_mask = combined.duplicated(subset=["date", "hour_ist"], keep=False)
    if dupe_mask.any():
        dupes = combined[dupe_mask]
        n_conflicting = dupes.groupby(["date", "hour_ist"])["aqi"].nunique().gt(1).sum()
        print(
            f"  {dupe_mask.sum()} duplicate (date, hour) readings found for "
            f"{station_row['station_id']} (likely re-downloaded under a new "
            f"filename convention) -- keeping one per hour."
            + (f" {n_conflicting} of them disagree on the AQI value!" if n_conflicting else "")
        )
        combined = combined.drop_duplicates(subset=["date", "hour_ist"], keep="first")

    return combined


def daily_aggregate(hourly_df):
    """[date, hour_ist, aqi] -> [date, AQI_mean, AQI_max, AQI_min, AQI_valid_hours, AQI_eod].

    IMPORTANT -- what the hourly numbers are: CPCB's hourly AQI is a *trailing 24-hour running
    average* ("AQI at 6am ... will incorporate data from 6am on previous day to the current day",
    airquality.cpcb.gov.in/ccr_docs/How_AQI_Calculated.pdf). So AQI_mean, the mean of 24 such
    values, blurs roughly two days: about half its weight is on the PREVIOUS day. It is kept for
    continuity, but AQI_eod is the value that actually matches the day:

      AQI_eod = the hourly AQI at IST 05:00 on the next IST day, i.e. the reading whose 24 h
      window ends ~23:30 UTC and therefore covers (almost exactly) the UTC day that the
      satellite / ERA5 daily values describe. (With ALIGN_TO_UTC_CALENDAR_DAY=False it is the
      23:00 IST reading instead, whose window is the IST calendar day.)

    Verified on this data (see docs/PROJECT_BRIEF.md, alignment audit): AQI_eod correlates more
    strongly with same-day satellite NO2 / wind than AQI_mean does and much less with the previous
    day's, and has lower day-to-day autocorrelation.
    """
    g = hourly_df.groupby("date")["aqi"]
    daily = g.agg(AQI_mean="mean", AQI_max="max", AQI_min="min", AQI_valid_hours="count").reset_index()
    eod = (hourly_df[hourly_df["hour_ist"] == EOD_HOUR_IST]
           .drop_duplicates("date").set_index("date")["aqi"].rename("AQI_eod"))
    daily = daily.merge(eod, left_on="date", right_index=True, how="left")
    return daily[daily["AQI_valid_hours"] >= MIN_VALID_HOURS]


def aqi_band(value):
    if pd.isna(value):
        return None
    for lo, hi, label in AQI_BANDS:
        if 0 <= value <= hi:
            return label
    return None


# ============================================================
# SPATIAL MATCH
# ============================================================

def nearest_grid_point(stations_df, grid_df):
    """For each station, find the nearest unique (latitude, longitude) in its city's grid."""
    matches = []
    for city, city_stations in stations_df.groupby("city"):
        city_grid = grid_df[grid_df["city"] == city]
        if city_grid.empty:
            print(f"  WARNING: no grid data for city '{city}' -- its stations will be dropped")
            continue

        coords = city_grid[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
        tree = BallTree(np.radians(coords[["latitude", "longitude"]].values), metric="haversine")

        query = np.radians(city_stations[["lat", "lon"]].values)
        dist, idx = tree.query(query, k=1)

        for (_, station), d, i in zip(city_stations.iterrows(), dist[:, 0], idx[:, 0]):
            matches.append(
                {
                    "station_id": station["station_id"],
                    "grid_lat": coords.loc[i, "latitude"],
                    "grid_lon": coords.loc[i, "longitude"],
                    "match_distance_km": d * EARTH_RADIUS_KM,
                }
            )
    return pd.DataFrame(matches)


# ============================================================
# MAIN
# ============================================================

def load_gee_grids():
    files = sorted(GEE_DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No grid CSVs found in {GEE_DATA_DIR}. Run gee_extract.py first.")
    frames = [pd.read_csv(f) for f in files]
    grid = pd.concat(frames, ignore_index=True)
    grid["date"] = pd.to_datetime(grid["date"])
    return grid


def main():
    stations = pd.read_csv(STATIONS_CSV)
    grid = load_gee_grids()

    print(f"Grid rows loaded : {len(grid)} across {grid['city'].nunique()} cities")
    print(f"Stations         : {len(stations)}")

    match = nearest_grid_point(stations, grid)
    stations = stations.merge(match, on="station_id", how="inner")

    too_far = stations[stations["match_distance_km"] > MAX_MATCH_DISTANCE_KM]
    if len(too_far):
        print(f"  Dropping {len(too_far)} station(s) farther than {MAX_MATCH_DISTANCE_KM} km from any grid point:")
        for _, r in too_far.iterrows():
            print(f"    {r['station_id']}: {r['match_distance_km']:.1f} km")
    stations = stations[stations["match_distance_km"] <= MAX_MATCH_DISTANCE_KM]

    all_rows = []
    for _, station in stations.iterrows():
        print(f"\n{station['station_id']} -> grid point ({station['grid_lat']:.4f}, {station['grid_lon']:.4f}), "
              f"{station['match_distance_km']:.2f} km away")

        hourly = load_station_hourly(station)
        if hourly.empty:
            continue
        daily = daily_aggregate(hourly)
        print(f"  {len(daily)} valid days of AQI (>= {MIN_VALID_HOURS} hours each)")

        station_grid = grid[
            (grid["city"] == station["city"])
            & (grid["latitude"] == station["grid_lat"])
            & (grid["longitude"] == station["grid_lon"])
        ]

        merged = daily.merge(station_grid, on="date", how="inner")
        if merged.empty:
            print("  WARNING: no overlapping dates between AQI and satellite/met data -- skipped")
            continue

        merged["station_id"] = station["station_id"]
        merged["station_name"] = station["station_name"]
        merged["station_lat"] = station["lat"]
        merged["station_lon"] = station["lon"]
        merged["grid_lat"] = station["grid_lat"]
        merged["grid_lon"] = station["grid_lon"]
        merged["match_distance_km"] = station["match_distance_km"]
        merged["AQI_band"] = merged[f"AQI_{DAILY_AGGREGATION}"].apply(aqi_band)
        merged["AQI_mean_band"] = merged["AQI_mean"].apply(aqi_band)
        # a sensor stuck at one value all day (not the legitimate 500 cap) -- suspect data
        merged["aqi_flat_day"] = (merged["AQI_min"] == merged["AQI_max"]) & (merged["AQI_max"] < 500)
        sat_cols = [c for c in SATELLITE_COLUMNS if c in merged.columns]
        merged["satellite_bands_present"] = merged[sat_cols].notna().sum(axis=1)

        all_rows.append(merged)
        print(f"  {len(merged)} aligned rows")

    if not all_rows:
        raise SystemExit("No aligned rows produced -- check station coordinates, data_dir paths, and date overlap.")

    result = pd.concat(all_rows, ignore_index=True)

    front_cols = [
        "date", "city", "climatic_zone", "station_id", "station_name",
        "station_lat", "station_lon", "grid_lat", "grid_lon", "match_distance_km",
        "AQI_eod", "AQI_mean", "AQI_max", "AQI_min", "AQI_valid_hours", "AQI_band", "aqi_flat_day", "satellite_bands_present",
    ]
    other_cols = [c for c in result.columns if c not in front_cols]
    result = result[front_cols + other_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(result)} rows to {OUTPUT_PATH}")
    print(result["city"].value_counts())


if __name__ == "__main__":
    main()
