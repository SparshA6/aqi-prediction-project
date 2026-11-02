import calendar, datetime as dt, glob, os, re
from collections import defaultdict
import numpy as np
import openpyxl
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT)
st = pd.read_csv("config/stations.csv")
al = pd.read_csv("data/processed/aligned_dataset.csv", parse_dates=["date"])
MONTHS = list(calendar.month_name)[1:]
IST = dt.timedelta(hours=5, minutes=30)

def ym_from_name(b):
    m = re.search(r"_(" + "|".join(MONTHS) + r")_(\d{4})_Hourly", b)          # new naming
    if m: return int(m.group(2)), MONTHS.index(m.group(1)) + 1
    m = re.search(r"_(\d{4})_(" + "|".join(MONTHS) + r")_", b)                 # old naming
    return int(m.group(1)), MONTHS.index(m.group(2)) + 1

hourly = {}          # station_id -> {(utc_date): [values]},  also raw (ist_datetime -> value)
raw_ist = {}
structure_issues = []
non_numeric = 0
for r in st.itertuples():
    ist_vals = {}
    for f in sorted(glob.glob(os.path.join(r.data_dir, "*.xlsx"))):
        y, mo = ym_from_name(os.path.basename(f))
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()
        if not rows: structure_issues.append((r.station_id, os.path.basename(f), "empty")); continue
        header, body = rows[0], rows[1:]
        hrs = [str(h)[:2] for h in header[1:]]
        if hrs != [f"{i:02d}" for i in range(24)]:
            structure_issues.append((r.station_id, os.path.basename(f), f"header hours {hrs[:3]}..")); continue
        ndays = calendar.monthrange(y, mo)[1]
        days = [row[0] for row in body if row[0] is not None]
        if days != list(range(1, ndays + 1)) and days != list(range(1, 32))[:len(days)]:
            structure_issues.append((r.station_id, os.path.basename(f), f"day rows {days[:2]}..{days[-1:]} vs {ndays} days"))
        for row in body:
            if row[0] is None: continue
            day = int(row[0])
            if day > ndays:
                if any(v not in (None, "") for v in row[1:]):
                    structure_issues.append((r.station_id, os.path.basename(f), f"values on nonexistent day {day}"))
                continue
            for h, v in enumerate(row[1:25]):
                if v is None or v == "": continue
                try: v = float(v)
                except Exception: non_numeric += 1; continue
                ist_vals[dt.datetime(y, mo, day, h)] = v      # duplicates (old+new files) collapse here
    raw_ist[r.station_id] = ist_vals

print("=== B1. file structure ===")
print("files with structural problems:", structure_issues[:10] or "none", f"(total {len(structure_issues)})")
print("non-numeric AQI cells:", non_numeric)

print("\n=== B2. independent recompute of AQI for EVERY aligned row (UTC-day binning) ===")
def daily_utc(vals):
    b = defaultdict(list)
    for ts, v in vals.items(): b[(ts - IST).date()].append(v)
    return {pd.Timestamp(d): (np.mean(v), max(v), len(v)) for d, v in b.items()}

rec = {sid: daily_utc(v) for sid, v in raw_ist.items()}
diff_mean = diff_max = diff_n = missing = 0
worst = 0.0
for r in al.itertuples():
    t = rec[r.station_id].get(r.date)
    if t is None: missing += 1; continue
    mean, mx, n = t
    worst = max(worst, abs(mean - r.AQI_mean))
    diff_mean += not np.isclose(mean, r.AQI_mean, atol=1e-6)
    diff_max += not np.isclose(mx, r.AQI_max)
    diff_n += (n != r.AQI_valid_hours)
print(f"aligned rows checked: {len(al):,} | not reproducible from raw: {missing}")
print(f"AQI_mean mismatches: {diff_mean} (worst abs diff {worst:.2e}) | AQI_max mismatches: {diff_max} | valid_hours mismatches: {diff_n}")

# rows the pipeline should have produced but didn't (AQI day valid + satellite/met grid day exists)?
print("\n=== B3. completeness: valid AQI days that have grid data but are missing from the aligned table ===")
have = set(zip(al.station_id, al.date))
missing_rows = 0; examples = []
gridcache = {}
for r in st.itertuples():
    if r.city not in gridcache:
        g = pd.read_csv(f"data/raw/gee/{r.city}.csv", usecols=["date", "latitude", "longitude"], parse_dates=["date"])
        gridcache[r.city] = g
    row0 = al[al.station_id == r.station_id]
    if row0.empty: continue
    gl, go = row0.iloc[0][["grid_lat", "grid_lon"]]
    g = gridcache[r.city]
    gdates = set(g[(np.isclose(g.latitude, gl)) & (np.isclose(g.longitude, go))].date)
    for d, (mean, mx, n) in rec[r.station_id].items():
        if n >= 12 and d in gdates and (r.station_id, d) not in have:
            missing_rows += 1
            if len(examples) < 5: examples.append((r.station_id, str(d.date()), n))
print("valid (>=12 h) AQI days with a grid row but absent from aligned table:", missing_rows, examples)
# and rows in the table that shouldn't be there
extra = sum(1 for r in al.itertuples() if rec[r.station_id].get(r.date, (0, 0, 0))[2] < 12)
print("aligned rows with < 12 valid hours (should be 0):", extra)
print("dates outside 2023 in aligned table:", int((al.date.dt.year != 2023).sum()), "| min/max:", al.date.min().date(), al.date.max().date())

print("\n=== B4. feature values: re-lookup in raw satellite/met files ===")
tot_bad = defaultdict(int); n = 0
for city, sub in al.groupby("city"):
    g = pd.read_csv(f"data/raw/gee/{city}.csv", parse_dates=["date"])
    m = sub.merge(g, left_on=["date", "grid_lat", "grid_lon"], right_on=["date", "latitude", "longitude"], how="left", suffixes=("", "_raw"), indicator=True)
    assert (m["_merge"] == "both").all(), f"{city}: aligned row without raw grid row"
    for c in ["NO2", "HCHO", "AOD", "temperature_2m", "dewpoint_temperature_2m", "surface_pressure", "total_precipitation_sum", "u_component_of_wind_10m", "v_component_of_wind_10m"]:
        a, b = m[c], m[c + "_raw"]
        bad = ~(np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))
        tot_bad[c] += int(bad.sum())
    n += len(m)
    sat = m[["NO2", "HCHO", "AOD"]].notna().sum(axis=1)
    tot_bad["satellite_bands_present"] += int((sat != m["satellite_bands_present"]).sum())
print(f"rows checked: {n:,} | mismatching values per column:", dict(tot_bad))
