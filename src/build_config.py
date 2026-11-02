"""
Generate config/cities.csv and add each city's stations to config/stations.csv,
from the all-India station metadata (data/raw/station_metadata/station_data_raw.csv,
originally huggingface.co/datasets/Zeel/C1).

Idempotent: cities.csv is rebuilt every time; a city that already has rows in
stations.csv is left untouched (so hand-curated rows/notes survive -- e.g. Delhi). To add a city, add it to
CITY_SPEC below and rerun:  python src/build_config.py

Choices worth knowing about:
  * The satellite grid bbox for a city covers ALL of that city's stations in the metadata
    (+ BBOX_PAD_DEG), so adding more of its stations later doesn't need a re-pull.
  * Cities with more than MAX_STATIONS_PER_CITY stations are thinned by greedy
    farthest-point sampling (max spatial spread), so one city (e.g. Mumbai, 30
    stations) doesn't dominate. Delhi is exempt (already fully downloaded).
  * climatic_zone labels are QUALITATIVE, assigned by us -- treat as a working
    assumption, not a formal Koppen/IMD classification.
"""

import re

import numpy as np
import pandas as pd

from paths import CITIES_CSV, ROOT, STATION_META_DIR, STATIONS_CSV

BBOX_PAD_DEG = 0.10
MAX_STATIONS_PER_CITY = 14
EXEMPT_FROM_CAP = {"Delhi"}

# city -> (portal state name, qualitative climatic zone)
CITY_SPEC = {
    "Delhi":     ("Delhi",         "IGP_plains_stagnation_prone"),
    "Lucknow":   ("Uttar Pradesh", "IGP_plains_stagnation_prone"),
    "Kolkata":   ("West Bengal",   "humid_eastern_delta"),
    "Mumbai":    ("Maharashtra",   "coastal_ventilated_west"),
    "Chennai":   ("Tamil Nadu",    "coastal_tropical_southeast"),
    "Bengaluru": ("Karnataka",     "upland_plateau_mild"),
    "Hyderabad": ("Telangana",     "semi_arid_deccan"),
    "Jaipur":    ("Rajasthan",     "semi_arid_dust"),
}

# Delhi's grid must stay exactly as pulled so the existing data/raw/gee/Delhi.csv remains valid.
FIXED_BBOX = {"Delhi": (76.70, 28.30, 77.50, 29.00)}


def load_metadata():
    df = pd.read_csv(STATION_META_DIR / "station_data_raw.csv")
    parts = df["station"].str.extract(r"^(?P<name>.*),\s*(?P<city>[^,]+?)\s*-\s*(?P<agency>[^-]+?)\s*$")
    return pd.concat([df, parts], axis=1).dropna(subset=["city", "latitude", "longitude"])


def slug(name, agency):
    return re.sub(r"[^A-Za-z0-9]", "", name) + "_" + re.sub(r"[^A-Za-z0-9]", "", agency)


def farthest_point_sample(coords, k):
    """Indices of k points with maximal spatial spread; starts from the point nearest the centroid."""
    pts = np.asarray(coords, dtype=float)
    if len(pts) <= k:
        return list(range(len(pts)))
    chosen = [int(np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))]
    dmin = ((pts - pts[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, ((pts - pts[nxt]) ** 2).sum(axis=1))
    return sorted(chosen)


def main():
    meta = load_metadata()
    stations = pd.read_csv(STATIONS_CSV)
    have = set(stations["station_name"])
    curated_cities = set(stations["city"])  # cities whose station list was already built/hand-edited: don't touch

    city_rows, new_station_rows = [], []
    for city, (state, zone) in CITY_SPEC.items():
        cm = meta[meta["city"] == city].reset_index(drop=True)
        if cm.empty:
            print(f"WARNING: no stations for {city} in metadata")
            continue

        # ignore obviously bad coordinates when sizing the grid (> 0.5 deg from the median)
        med_lat, med_lon = cm["latitude"].median(), cm["longitude"].median()
        ok = ((cm["latitude"] - med_lat).abs() < 0.5) & ((cm["longitude"] - med_lon).abs() < 0.5)
        if (~ok).any():
            print(f"{city}: ignoring {int((~ok).sum())} outlier station(s): {cm.loc[~ok, 'station'].tolist()}")
        cm = cm[ok].reset_index(drop=True)

        if city in FIXED_BBOX:
            min_lon, min_lat, max_lon, max_lat = FIXED_BBOX[city]
        else:
            min_lon, max_lon = cm["longitude"].min() - BBOX_PAD_DEG, cm["longitude"].max() + BBOX_PAD_DEG
            min_lat, max_lat = cm["latitude"].min() - BBOX_PAD_DEG, cm["latitude"].max() + BBOX_PAD_DEG

        city_rows.append({
            "city": city, "portal_state": state, "portal_city": city, "climatic_zone": zone,
            "min_lon": round(min_lon, 2), "min_lat": round(min_lat, 2),
            "max_lon": round(max_lon, 2), "max_lat": round(max_lat, 2),
            "n_stations_in_metadata": len(cm),
        })

        keep = cm
        if city not in EXEMPT_FROM_CAP and len(cm) > MAX_STATIONS_PER_CITY:
            idx = farthest_point_sample(cm[["latitude", "longitude"]].values, MAX_STATIONS_PER_CITY)
            keep = cm.iloc[idx]
            print(f"{city}: {len(cm)} stations in metadata -> keeping {len(keep)} (max spatial spread)")

        if city in curated_cities:
            continue

        for r in keep.itertuples():
            if r.station in have:
                continue
            s = slug(r.name, r.agency)
            new_station_rows.append({
                "station_id": f"{s.lower()}_{city.lower()}",
                "station_name": r.station,
                "city": city,
                "climatic_zone": zone,
                "lat": r.latitude,
                "lon": r.longitude,
                "data_dir": f"data/raw/cpcb/{city}/{s}",
                "notes": "Coordinates from huggingface.co/datasets/Zeel/C1 raw CSV; not independently verified. AQI not yet downloaded.",
            })

    cities = pd.DataFrame(city_rows)
    cities.to_csv(CITIES_CSV, index=False)
    print(f"\nwrote {CITIES_CSV.relative_to(ROOT)} ({len(cities)} cities)")

    if new_station_rows:
        out = pd.concat([stations, pd.DataFrame(new_station_rows)], ignore_index=True)
        assert out["station_id"].is_unique, out[out["station_id"].duplicated(keep=False)]
        assert out["data_dir"].is_unique, out[out["data_dir"].duplicated(keep=False)]
        out.to_csv(STATIONS_CSV, index=False)
        print(f"added {len(new_station_rows)} stations to {STATIONS_CSV.relative_to(ROOT)} (now {len(out)})")
    else:
        print("no new stations to add")

    print("\n", cities.to_string(index=False))


if __name__ == "__main__":
    main()
