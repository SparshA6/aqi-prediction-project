import glob, os, re, math, sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT)
st = pd.read_csv("config/stations.csv")
cities = pd.read_csv("config/cities.csv").set_index("city")
al = pd.read_csv("data/processed/aligned_dataset.csv", parse_dates=["date"])

def hav(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

print("=== A1. station coordinates sane? ===")
bad = []
for r in st.itertuples():
    c = cities.loc[r.city]
    inside = (c.min_lon <= r.lon <= c.max_lon) and (c.min_lat <= r.lat <= c.max_lat)
    if not inside:
        bad.append((r.station_name, "outside city bbox", r.lat, r.lon))
print("stations outside their city's grid bbox:", bad or "none")
cen = st.groupby("city")[["lat", "lon"]].median()
st["d_center_km"] = [hav(r.lat, r.lon, cen.loc[r.city, "lat"], cen.loc[r.city, "lon"]) for r in st.itertuples()]
print("distance from city median (km): max per city")
print(st.groupby("city")["d_center_km"].max().round(1).to_string())
dupc = st[st.duplicated(["lat", "lon"], keep=False)]
print("stations sharing identical coordinates:", dupc[["station_name"]].values.ravel().tolist() or "none")
# near-duplicate (<150 m) different stations
close = []
for i, a in st.iterrows():
    for j, b in st.iterrows():
        if j <= i or a.city != b.city: continue
        d = hav(a.lat, a.lon, b.lat, b.lon)
        if d < 0.15: close.append((a.station_name, b.station_name, round(d * 1000)))
print("station pairs < 150 m apart:", close or "none")

print("\n=== A2. does each folder contain files of the RIGHT station? ===")
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
problems, site_ids = [], {}
for r in st.itertuples():
    files = glob.glob(os.path.join(r.data_dir, "*.xlsx"))
    sids = set()
    for f in files:
        b = os.path.basename(f)
        m = re.search(r"Site_(\d+)_(.*?)_(?:January|February|March|April|May|June|July|August|September|October|November|December)_", b)
        if not m:
            problems.append((r.station_id, "unparsable filename", b)); continue
        sids.add(m.group(1))
        # station-name tokens in the file name must match the configured station's base name
        base = norm(re.split(r",\s*" + re.escape(r.city), r.station_name)[0])
        fname_norm = norm(m.group(2))
        if base not in fname_norm and fname_norm not in norm(r.station_name):
            problems.append((r.station_id, "name mismatch", b))
    if len(sids) != 1: problems.append((r.station_id, f"site ids in folder: {sorted(sids)}", ""))
    for s in sids: site_ids.setdefault(s, []).append(r.station_id)
print("folders with a wrong/inconsistent station:", problems or "none")
shared = {k: v for k, v in site_ids.items() if len(v) > 1}
print("site ids used by more than one configured station:", shared or "none")
print("files per station (should be 12):", st.assign(n=[len(glob.glob(os.path.join(d, '*.xlsx'))) for d in st.data_dir]).query("n != 12")[["station_id", "n"]].values.tolist() or "all 12  (Alipur has 24 = old+new naming duplicates)")

print("\n=== A3. nearest-grid-cell match: brute-force recompute ===")
worst = 0; mism = []
for city, sub in st.groupby("city"):
    g = pd.read_csv(f"data/raw/gee/{city}.csv", usecols=["latitude", "longitude"]).drop_duplicates().reset_index(drop=True)
    for r in sub.itertuples():
        d = hav(r.lat, r.lon, g.latitude.values, g.longitude.values)
        k = int(np.argmin(d))
        rows = al[al.station_id == r.station_id]
        if rows.empty: continue
        gl, go, md = rows.iloc[0][["grid_lat", "grid_lon", "match_distance_km"]]
        if not (math.isclose(gl, g.latitude[k], abs_tol=1e-9) and math.isclose(go, g.longitude[k], abs_tol=1e-9)):
            mism.append((r.station_id, (gl, go), (g.latitude[k], g.longitude[k])))
        worst = max(worst, abs(md - d[k]))
        if rows.station_lat.nunique() != 1 or rows.grid_lat.nunique() != 1: mism.append((r.station_id, "grid/coords vary within station"))
        if not (math.isclose(rows.iloc[0].station_lat, r.lat) and math.isclose(rows.iloc[0].station_lon, r.lon)): mism.append((r.station_id, "aligned coords differ from stations.csv"))
print("stations whose matched cell differs from brute-force nearest:", mism or "none")
print(f"max |reported distance - recomputed distance| = {worst:.6f} km")
print("\nstations with NO rows in aligned table:", sorted(set(st.station_id) - set(al.station_id)) or "none")
