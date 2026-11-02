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
        hourly[r.station_id] = pd.concat(parts).drop_duplicates("ts").set_index("ts")["aqi"].sort_index()

met = {}
for city in cell.city.unique():
    g = pd.read_csv(f"data/raw/gee/{city}.csv", parse_dates=["date"])
    for sid in cell[cell.city == city].index:
        gl, go = cell.loc[sid, ["grid_lat", "grid_lon"]]
        m = g[(np.isclose(g.latitude, gl)) & (np.isclose(g.longitude, go))].set_index("date").sort_index()
        m["wspd"] = np.hypot(m.u_component_of_wind_10m, m.v_component_of_wind_10m)
        m["precip"] = m.total_precipitation_sum * 1000
        met[sid] = m[["wspd", "precip", "NO2", "temperature_2m"]]

def targets(s, days):
    """daily target series for UTC-days `days`, several definitions"""
    def window_mean(lo, hi, minn=12):
        vals = []
        for d in days:
            w = s.reindex(d + pd.to_timedelta(np.arange(lo, hi + 1), unit="h")).dropna()
            vals.append(w.mean() if len(w) >= minn else np.nan)
        return np.array(vals)
    at = lambda off: s.reindex(days + pd.Timedelta(hours=off)).values
    return pd.DataFrame({
        "T1 mean of UTC-day bin (current)": window_mean(6, 29),
        "T2 mean of IST calendar day (naive)": window_mean(0, 23),
        "T3 AQI at 05:00 IST next day (24h window = UTC day)": at(29),
        "T4 AQI at 06:00 IST next day": at(30),
        "T5 AQI at 23:00 IST same day (24h window = IST day)": at(23),
    }, index=days)

frames = []
for sid, s in hourly.items():
    days = met[sid].index
    t = targets(s, days)
    t["sid"] = sid
    frames.append(t.join(met[sid]).reset_index().rename(columns={"index": "d", "date": "d"}))
big = pd.concat(frames)
big["month"] = big.d.dt.month
tcols = [c for c in big.columns if c.startswith("T")]

def anom(df, cols):
    df = df.copy()
    for c in cols: df[c] = df[c] - df.groupby(["sid", "month"])[c].transform("mean")
    return df

print("rows:", len(big))
# pairwise corr with met of day d+lag
rows = []
for lag in (-1, 0, 1):
    mm = big[["sid", "d", "wspd", "precip", "NO2"]].copy()
    mm["d"] = mm["d"] - pd.Timedelta(days=lag)          # so that merge on d gives met(d+lag)
    mm = mm.rename(columns={"wspd": "wspd_l", "precip": "precip_l", "NO2": "NO2_l"})
    x = big.merge(mm, on=["sid", "d"])
    x = anom(x, tcols + ["wspd_l", "precip_l", "NO2_l"])
    for t in tcols:
        rows.append({"target": t, "met day": f"d{lag:+d}" if lag else "d (same day)",
                     "wspd": x[t].corr(x.wspd_l), "precip": x[t].corr(x.precip_l), "NO2 (sat.)": x[t].corr(x.NO2_l)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
for var in ("wspd", "precip", "NO2 (sat.)"):
    p = res.pivot(index="target", columns="met day", values=var)[["d-1", "d (same day)", "d+1"]]
    print(f"\ncorrelation with {var} (anomalies, pooled):")
    print(p.round(3).to_string())

# smoothness: lag-1 autocorrelation of anomalies within station
big = big.sort_values(["sid", "d"])
ac = {}
a = anom(big, tcols)
for t in tcols:
    x = a[t]; y = a.groupby("sid")[t].shift(1); ok = a.groupby("sid")["d"].diff().eq(pd.Timedelta(days=1))
    ac[t] = x[ok].corr(y[ok])
print("\nlag-1 day autocorrelation of each daily target (anomalies):")
print(pd.Series(ac).round(3).to_string())
