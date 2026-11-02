"""
Pull satellite pollution retrievals + ERA5-Land meteorology from Google Earth
Engine, sampled on a grid over each configured city, for a date range.

Sources:
    NO2, HCHO      Sentinel-5P TROPOMI L3 (COPERNICUS/S5P/OFFL/...)
    AOD            MODIS MAIAC (MODIS/061/MCD19A2_GRANULES)
    meteorology    ERA5-Land daily aggregates (ECMWF/ERA5_LAND/DAILY_AGGR)
    CO, SO2, O3    Sentinel-5P L3, optional (INCLUDE_EXTRA_POLLUTANTS)

Output: one CSV per city in data/raw/gee, one row per grid point per day (a day is
kept whenever the ERA5 weather exists; cloud-masked satellite values are NaN), columns
    date, latitude, longitude, city, climatic_zone,
    NO2, HCHO, AOD, temperature_2m, dewpoint_temperature_2m,
    surface_pressure, total_precipitation_sum,
    u_component_of_wind_10m, v_component_of_wind_10m
(plus CO/SO2/O3 if enabled, plus boundary_layer_height if enabled -- see the
caveat on that below).

Setup (one-time):
    pip install earthengine-api pandas tqdm
    earthengine authenticate
        -- opens a browser; needs a Google Cloud project with the Earth
           Engine API enabled: https://code.earthengine.google.com/register

Usage:
    1. EE_PROJECT below; cities live in config/cities.csv.
    2. python src/gee_extract.py --dry-run               # grid size / date count, no EE calls
    3. python src/gee_extract.py [--city Lucknow Jaipur]  # the real pull -> data/raw/gee/<City>.csv

A run can take a long time (one Earth Engine call per date per grid-chunk,
per city) and can be interrupted safely -- it writes each city's CSV
incrementally and skips dates it has already written if you rerun it.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from paths import CITIES_CSV, GEE_DIR

# ============================================================
# CONFIGURATION
# ============================================================

EE_PROJECT = "aqi-prediction-505817"  # registered Earth Engine cloud project

OUTPUT_DIR = GEE_DIR

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"  # inclusive

GRID_SPACING_DEG = 0.03  # ~3 km between sample points -- tune per city size / EE quota

INCLUDE_EXTRA_POLLUTANTS = False  # Sentinel-5P CO, SO2, O3

# Boundary-layer height is NOT part of ERA5-Land (it's an atmospheric, not
# land-surface, variable). GEE's curated ERA5 mirrors don't reliably expose
# it either -- verify against the current Earth Engine Data Catalog
# (https://developers.google.com/earth-engine/datasets) before enabling
# this, or pull BLH separately via the Copernicus Climate Data Store (CDS)
# API and join it in during combine_with_aqi.py instead.
INCLUDE_BOUNDARY_LAYER_HEIGHT = False
ERA5_BLH_COLLECTION = "ECMWF/ERA5/DAILY"  # UNVERIFIED -- confirm band exists before use
ERA5_BLH_BAND = "boundary_layer_height"



def load_cities(csv_path=CITIES_CSV):
    """config/cities.csv -> {city: {"bbox": (min_lon, min_lat, max_lon, max_lat), "climatic_zone": ...}}

    Add or remove cities by editing that file (config/build_config.py can generate rows
    from the station metadata). Changing a city's bbox or GRID_SPACING_DEG after data
    has been pulled changes its grid -- delete that city's CSV in data/raw/gee first.
    """
    df = pd.read_csv(csv_path)
    return {
        r.city: {
            "bbox": (r.min_lon, r.min_lat, r.max_lon, r.max_lat),
            "climatic_zone": r.climatic_zone,
        }
        for r in df.itertuples()
    }


CITIES = load_cities()

SAMPLE_SCALE_M = 1000  # nominal pixel scale (m) Earth Engine samples at
POINTS_PER_REQUEST = 1500  # grid points per sampleRegions() call
RETRY_LIMIT = 4
RETRY_WAIT_SEC = 15

# ============================================================
# EARTH ENGINE BAND DEFINITIONS
# ============================================================

S5P_NO2 = ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", "NO2")
S5P_HCHO = ("COPERNICUS/S5P/OFFL/L3_HCHO", "tropospheric_HCHO_column_number_density", "HCHO")
S5P_CO = ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density", "CO")
S5P_SO2 = ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", "SO2")
S5P_O3 = ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density", "O3")

MODIS_AOD_COLLECTION = "MODIS/061/MCD19A2_GRANULES"
MODIS_AOD_BAND = "Optical_Depth_047"
MODIS_AOD_SCALE = 0.001  # per MCD19A2 documentation

ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_LAND_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

# Value Earth Engine gets for cloud-masked satellite pixels (turned into NaN when writing the CSV).
# Far outside any physical value of NO2/HCHO/AOD (HCHO can be slightly negative, so 0 or -1 won't do).
SATELLITE_MASK_VALUE = -9999
SATELLITE_COLUMNS = ["NO2", "HCHO", "AOD"] + (["CO", "SO2", "O3"] if INCLUDE_EXTRA_POLLUTANTS else [])

OUTPUT_COLUMNS = (
    ["date", "latitude", "longitude", "city", "climatic_zone", "NO2", "HCHO", "AOD"]
    + (["CO", "SO2", "O3"] if INCLUDE_EXTRA_POLLUTANTS else [])
    + ERA5_LAND_BANDS
    + (["boundary_layer_height"] if INCLUDE_BOUNDARY_LAYER_HEIGHT else [])
)


# ============================================================
# GRID
# ============================================================

def build_grid(bbox, spacing_deg):
    """Regular lat/lon grid of points covering bbox = (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lats, lons = [], []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            lats.append(round(lat, 6))
            lons.append(round(lon, 6))
            lon += spacing_deg
        lat += spacing_deg
    return list(zip(lats, lons))


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ============================================================
# EARTH ENGINE
# ============================================================

def init_earth_engine():
    import ee

    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    return ee


def daily_composite(ee, date_str):
    """One ee.Image per day: all pollutant + met bands, mosaicked over [date, date+1)."""
    d = ee.Date(date_str)
    d_next = d.advance(1, "day")

    def band(collection_id, band_name, out_name, scale=1.0, satellite=True):
        collection = ee.ImageCollection(collection_id).filterDate(d, d_next).select(band_name)
        # An empty daily collection has zero bands. Retain a named, masked
        # placeholder so missing satellite coverage does not erase weather rows.
        empty = ee.Image.constant(0).rename(band_name).updateMask(ee.Image.constant(0))
        img = ee.Image(ee.Algorithms.If(collection.size().gt(0), collection.mean(), empty))
        if scale != 1.0:
            img = img.multiply(scale)
        img = img.rename(out_name)
        if satellite:
            # Cloud-masked pixels become a sentinel instead of dropping the whole grid point
            # (sampleRegions drops a point if ANY band is masked, which used to throw away
            # the always-available ERA5 weather on every cloudy day). Sentinel -> NaN in run().
            img = img.unmask(SATELLITE_MASK_VALUE)
        return img

    img = band(*S5P_NO2[:2], S5P_NO2[2])
    img = img.addBands(band(*S5P_HCHO[:2], S5P_HCHO[2]))

    if INCLUDE_EXTRA_POLLUTANTS:
        for coll, bname, out in (S5P_CO, S5P_SO2, S5P_O3):
            img = img.addBands(band(coll, bname, out))

    img = img.addBands(band(MODIS_AOD_COLLECTION, MODIS_AOD_BAND, "AOD", MODIS_AOD_SCALE))

    era5 = ee.ImageCollection(ERA5_LAND_DAILY).filterDate(d, d_next).select(ERA5_LAND_BANDS).mean()
    img = img.addBands(era5)

    if INCLUDE_BOUNDARY_LAYER_HEIGHT:
        img = img.addBands(band(ERA5_BLH_COLLECTION, ERA5_BLH_BAND, "boundary_layer_height", satellite=False))

    return img


def sample_points(ee, image, points, scale=SAMPLE_SCALE_M):
    """Sample `image` at a chunk of (lat, lon) points. Returns a list of dicts."""
    features = [
        ee.Feature(ee.Geometry.Point([lon, lat]), {"latitude": lat, "longitude": lon})
        for lat, lon in points
    ]
    fc = ee.FeatureCollection(features)
    sampled = image.sampleRegions(collection=fc, scale=scale, geometries=False)

    last_err = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            info = sampled.getInfo()
            return [f["properties"] for f in info["features"]]
        except Exception as e:  # Earth Engine transient errors, quota, timeouts
            last_err = e
            print(f"    retry {attempt}/{RETRY_LIMIT} after error: {e}")
            time.sleep(RETRY_WAIT_SEC)
    raise RuntimeError(f"sample_points failed after {RETRY_LIMIT} retries: {last_err}")


# ============================================================
# MAIN
# ============================================================

def already_done_dates(csv_path):
    """Dates already written to csv_path, normalized to YYYY-MM-DD strings.

    Compares by parsed date, not raw string, so a pre-existing file using a
    different date format (or containing rows from a different source/grid
    entirely) can't silently fail to match and get duplicated on top of.
    """
    if not csv_path.exists():
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=["date"])
        parsed = pd.to_datetime(existing["date"], errors="coerce")
        if parsed.isna().any():
            print(f"  WARNING: {parsed.isna().sum()} unparseable date(s) in {csv_path} -- "
                  f"those rows won't be recognized as already done.")
        return set(parsed.dropna().dt.strftime("%Y-%m-%d").unique())
    except Exception as e:
        print(f"  WARNING: couldn't read existing dates from {csv_path} ({e}) -- "
              f"treating as if no dates are done yet. Check this file isn't corrupted "
              f"before letting this run append to it.")
        return set()


def assert_grid_matches(csv_path, grid):
    """Refuse to append to a file whose grid points differ from the current config's grid.

    (The resume logic only looks at dates, so a changed bbox / spacing would otherwise
    silently mix two different grids in one file.)
    """
    if not csv_path.exists():
        return
    existing = pd.read_csv(csv_path, usecols=["latitude", "longitude"]).drop_duplicates()
    existing_pts = set(zip(existing["latitude"].round(6), existing["longitude"].round(6)))
    stray = existing_pts - set(grid)
    if stray:
        raise SystemExit(
            f"{csv_path} contains {len(stray)} grid point(s) that are not in the current grid "
            f"(bbox or GRID_SPACING_DEG changed?). Delete or move that file to re-pull it cleanly."
        )


def run(dry_run=False, only_cities=None):
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = CITIES
    if only_cities:
        unknown = [c for c in only_cities if c not in CITIES]
        if unknown:
            sys.exit(f"Unknown city/cities {unknown}. Known: {list(CITIES)}")
        selected = {c: CITIES[c] for c in only_cities}

    for city, cfg in selected.items():
        grid = build_grid(cfg["bbox"], GRID_SPACING_DEG)
        out_path = OUTPUT_DIR / f"{city}.csv"

        print(f"\n=== {city} ===")
        print(f"grid points : {len(grid)}")
        print(f"dates       : {len(date_range)} ({START_DATE} .. {END_DATE})")
        print(f"requests    : ~{len(date_range) * -(-len(grid) // POINTS_PER_REQUEST)}")
        print(f"output      : {out_path}")

        if dry_run:
            continue

        assert_grid_matches(out_path, grid)
        ee = init_earth_engine()
        done_dates = already_done_dates(out_path)
        header_written = out_path.exists() and len(done_dates) > 0

        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            if date_str in done_dates:
                continue

            print(f"  {date_str} ...", end=" ", flush=True)
            image = daily_composite(ee, date_str)

            rows = []
            for chunk in chunks(grid, POINTS_PER_REQUEST):
                rows.extend(sample_points(ee, image, chunk))

            if not rows:
                print("no data (cloud/gap) -- skipped")
                continue

            df = pd.DataFrame(rows)
            df[SATELLITE_COLUMNS] = df[SATELLITE_COLUMNS].replace(SATELLITE_MASK_VALUE, float("nan"))
            df["date"] = date_str
            df["city"] = city
            df["climatic_zone"] = cfg["climatic_zone"]
            df = df.reindex(columns=OUTPUT_COLUMNS)

            df.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True
            print(f"{len(df)} rows written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print grid/date sizes, no Earth Engine calls")
    parser.add_argument("--city", nargs="+", help="only these cities (names as in config/cities.csv); "
                                                  "lets several processes pull different cities in parallel")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    START_DATE, END_DATE, OUTPUT_DIR = args.start_date, args.end_date, args.output_dir
    if pd.Timestamp(START_DATE) > pd.Timestamp(END_DATE):
        parser.error("Start date must not be after end date")

    if not args.dry_run and EE_PROJECT == "your-gcp-project-id":
        sys.exit("Set EE_PROJECT at the top of this file to your Earth Engine cloud project id first.")

    run(dry_run=args.dry_run, only_cities=args.city)
