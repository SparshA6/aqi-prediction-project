import calendar, glob, os, re
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT)
st = pd.read_csv("config/stations.csv")
al = pd.read_csv("data/processed/aligned_dataset.csv", parse_dates=["date"])
cell = al.drop_duplicates("station_id").set_index("station_id")[["grid_lat", "grid_lon", "city"]]
MONTHS = list(calendar.month_name)[1:]

def ym(b):
    m = re.search(r"_(" + "|".join(MONTHS) + r")_(\d{4})_Hourly", b)
    if m: return int(m.group(2)), MONTHS.index(m.group(1)) + 1
    m = re.search(r"_(\d{4})_(" + "|".join(MONTHS) + r")_", b)
    return int(m.group(1)), MONTHS.index(m.group(2)) + 1

# ---- hourly AQI indexed by IST timestamp, per station
hourly = {}
for r in st.itertuples():
    parts = []
    for f in glob.glob(os.path.join(r.data_dir, "*.xlsx")):
        y, mo = ym(os.path.basename(f))
        d = pd.read_excel(f, header=0)
        if d.empty: continue
        d = d.rename(columns={d.columns[0]: "day"}).dropna(subset=["day"])
        long = d.melt(id_vars="day", var_name="h", value_name="aqi")
        long["aqi"] = pd.to_numeric(long["aqi"], errors="coerce")
        long = long.dropna(subset=["aqi"])
        long["ts"] = pd.to_datetime(dict(year=y, month=mo, day=long["day"].astype(int), hour=long["h"].astype(str).str[:2].astype(int)), errors="coerce")
        parts.append(long[["ts", "aqi"]].dropna())
    if parts:
        s = pd.concat(parts).drop_duplicates("ts").set_index("ts")["aqi"].sort_index()
        hourly[r.station_id] = s

# ---- daily met (UTC day) at each station's matched grid cell
met = {}
for city in cell.city.unique():
    g = pd.read_csv(f"data/raw/gee/{city}.csv", parse_dates=["date"])
    for sid in cell[cell.city == city].index:
        gl, go = cell.loc[sid, ["grid_lat", "grid_lon"]]
        m = g[(np.isclose(g.latitude, gl)) & (np.isclose(g.longitude, go))].set_index("date").sort_index()
        m["wspd"] = np.hypot(m.u_component_of_wind_10m, m.v_component_of_wind_10m)
        m["precip_mm"] = m.total_precipitation_sum * 1000
        m["temp_c"] = m.temperature_2m - 273.15
        met[sid] = m[["wspd", "precip_mm", "temp_c", "NO2", "surface_pressure"]]

def anomalies(df, cols):
    """remove station x month means -> keep day-to-day variation only"""
    key = [df["sid"], df["d"].dt.month]
    for c in cols:
        df[c] = df[c] - df.groupby(key)[c].transform("mean")
    return df

# ---- scan: AQI value at IST timestamp (UTC-day d + offset hours), correlated with met of UTC day d
rows = []
offsets = np.arange(0, 48)                     # hours after 00:00 IST of IST-date d
for sid, s in hourly.items():
    days = met[sid].index
    for off in offsets:
        ts = days + pd.Timedelta(hours=int(off))
        v = s.reindex(ts).values
        rows.append(pd.DataFrame({"sid": sid, "d": days, "off": off, "aqi": v}))
scan = pd.concat(rows).dropna(subset=["aqi"])
scan = scan.merge(pd.concat({k: v for k, v in met.items()}, names=["sid", "d"]).reset_index(), on=["sid", "d"])
res = []
for lag in (-1, 0, 1):
    pass
out = []
for off, g in scan.groupby("off"):
    g = anomalies(g.copy(), ["aqi", "wspd", "precip_mm", "temp_c"])
    out.append({"off_h_after_00IST_of_day_d": off,
                "window_end_UTC_hour": off - 5.5,
                "corr_wspd": g.aqi.corr(g.wspd), "corr_precip": g.aqi.corr(g.precip_mm), "corr_temp": g.aqi.corr(g.temp_c), "n": len(g)})
sc = pd.DataFrame(out)
pd.set_option("display.width", 200)
print("Correlation of AQI-at-timestamp with the weather of UTC-day d (anomalies; more negative = stronger)")
print("window_end_UTC_hour: 24 means the trailing-24h window ends exactly when UTC day d ends\n")
print(sc.round(3).to_string(index=False))
sc.to_csv("logs/hour_scan.csv", index=False)
best = sc.loc[sc.corr_wspd.idxmin()]
print("\nstrongest (most negative) wind-speed correlation at offset", int(best.off_h_after_00IST_of_day_d), "h  -> window ends", best.window_end_UTC_hour, "UTC-hours after the start of day d; corr =", round(best.corr_wspd, 3))
