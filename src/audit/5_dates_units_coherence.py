import os, sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT); sys.path.insert(0, "src")
pd.set_option("display.width", 220)

print("=== C4. Earth Engine date semantics (are 'date' D really the UTC day D?) ===")
import gee_extract as ge
ee = ge.init_earth_engine()
import datetime as dt
def t(ms): return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
for d in ("2023-01-15", "2023-07-15"):
    d0 = ee.Date(d); d1 = d0.advance(1, "day")
    era = ee.ImageCollection(ge.ERA5_LAND_DAILY).filterDate(d0, d1)
    ts = era.aggregate_array("system:time_start").getInfo()
    print(f"{d}: ERA5-Land daily images: {len(ts)}, time_start UTC = {[t(x) for x in ts]}")
    for name, coll in (("S5P NO2", ge.S5P_NO2[0]), ("S5P HCHO", ge.S5P_HCHO[0]), ("MODIS AOD", ge.MODIS_AOD_COLLECTION)):
        ts = ee.ImageCollection(coll).filterDate(d0, d1).aggregate_array("system:time_start").getInfo()
        hrs = sorted(int(t(x)[11:13]) for x in ts)
        print(f"   {name}: {len(ts)} images, UTC hours of acquisition {hrs[0]:02d}..{hrs[-1]:02d}  (all within the UTC day: {all(d == t(x)[:10] for x in ts)})")

print("\n=== D1. value ranges / units per city (satellite+met) ===")
rows = []
for city in pd.read_csv("config/cities.csv").city:
    g = pd.read_csv(f"data/raw/gee/{city}.csv")
    w = np.hypot(g.u_component_of_wind_10m, g.v_component_of_wind_10m)
    rows.append({"city": city,
                 "NO2 min/max (mol/m2)": f"{g.NO2.min():.1e}/{g.NO2.max():.1e}",
                 "HCHO min/max": f"{g.HCHO.min():.1e}/{g.HCHO.max():.1e}",
                 "AOD min/max": f"{g.AOD.min():.2f}/{g.AOD.max():.2f}",
                 "T2m C": f"{g.temperature_2m.min()-273.15:.0f}..{g.temperature_2m.max()-273.15:.0f}",
                 "dewpt>T rows": int((g.dewpoint_temperature_2m > g.temperature_2m + 0.01).sum()),
                 "pressure kPa": f"{g.surface_pressure.min()/1000:.0f}..{g.surface_pressure.max()/1000:.0f}",
                 "precip<0": int((g.total_precipitation_sum < 0).sum()),
                 "max wind m/s": round(w.max(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
dl = pd.read_csv("data/raw/gee/Delhi.csv", parse_dates=["date"])
mm = dl.groupby(dl.date.dt.month).temperature_2m.mean() - 273.15
print("Delhi mean 2m temperature by month (C):", mm.round(0).astype(int).tolist(), " <- expect ~14,18,24,30,34,35,32,31,30,26,20,15 (IMD climatology)")

print("\n=== D2. AQI value sanity ===")
al = pd.read_csv("data/processed/aligned_dataset.csv", parse_dates=["date"])
print("AQI_mean range:", al.AQI_mean.min().round(1), "-", al.AQI_mean.max().round(1), "| AQI_max range:", al.AQI_max.min(), "-", al.AQI_max.max(), "| rows with AQI_max > 500:", int((al.AQI_max > 500).sum()), "| rows AQI_mean == 0:", int((al.AQI_mean == 0).sum()))
print("rows with fewer than 24 valid hours:", f"{(al.AQI_valid_hours < 24).mean():.1%}", "| < 18 hours:", f"{(al.AQI_valid_hours < 18).mean():.1%}")
# flat-lined sensor days: max == min over the day
print("rows where the daily AQI is constant all day (max==mean):", int((al.AQI_max == al.AQI_mean).sum()))
# repeated identical daily means for many consecutive days (stuck sensor)
al = al.sort_values(["station_id", "date"])
run = (al.groupby("station_id")["AQI_mean"].diff().eq(0)).sum()
print("consecutive-day identical AQI_mean occurrences:", int(run))

print("\n=== D3. spatial coherence: does each station's daily series agree with its city-mates? ===")
piv = al.pivot_table(index="date", columns="station_id", values="AQI_mean")
city = al.drop_duplicates("station_id").set_index("station_id").city
lat = al.drop_duplicates("station_id").set_index("station_id")[["station_lat", "station_lon"]]
res = []
for c in city.unique():
    ids = [i for i in piv.columns if city[i] == c]
    an = piv[ids] - piv[ids].groupby(piv.index.month).transform("mean")
    for i in ids:
        others = [j for j in ids if j != i]
        cm = an[others].median(axis=1)
        ok = an[i].notna() & cm.notna()
        res.append({"city": c, "station": i, "n_days": int(ok.sum()), "corr_with_city_median": an[i][ok].corr(cm[ok]) if ok.sum() > 30 else np.nan})
r = pd.DataFrame(res)
print(r.groupby("city").corr_with_city_median.agg(["min", "median", "max"]).round(2).to_string())
print("\nlowest-agreement stations:")
print(r.sort_values("corr_with_city_median").head(8).round(2).to_string(index=False))
