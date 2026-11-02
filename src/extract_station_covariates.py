"""Extract pre-study elevation and land-cover covariates in station buffers."""

import json
from pathlib import Path
import ee
import pandas as pd
from paths import ROOT, STATIONS_CSV


def main():
    ee.Initialize(project="aqi-prediction-505817")
    stations = pd.read_csv(STATIONS_CSV)
    features = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(5000), {"station_id": r.station_id}) for r in stations.itertuples()]
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation_mean_5km_m")
    cover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    image = elevation
    for name, code in [("built_fraction_5km",50),("tree_fraction_5km",10),("crop_fraction_5km",40),("water_fraction_5km",80)]:
        image = image.addBands(cover.eq(code).rename(name).toFloat())
    values = image.reduceRegions(collection=ee.FeatureCollection(features), reducer=ee.Reducer.mean(), scale=100, tileScale=4).getInfo()
    df = pd.DataFrame([f["properties"] for f in values["features"]])
    if len(df) != len(stations) or df.station_id.duplicated().any() or df.isna().any().any():
        raise ValueError("Incomplete static covariates; retain missingness instead of inventing values")
    for c in df.columns:
        if "fraction" in c and not df[c].between(0,1).all():
            raise ValueError(f"Invalid fraction: {c}")
    out = ROOT / "data/processed/station_covariates"
    out.mkdir(parents=True, exist_ok=False)
    df.to_csv(out / "covariates.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(buffer_m=5000, sampling_scale_m=100,
        sources={"elevation":"USGS/SRTMGL1_003 (2000)","landcover":"ESA/WorldCover/v200 (2021)"},
        interpretation="Mean elevation and approximate land-cover fractions from 100m sampling inside 5km station buffers",
        source_links=["https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003","https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200"]),indent=2),encoding="utf-8")
    print(f"Static covariates saved for {len(df)} stations: {out}")


if __name__ == "__main__":
    main()
