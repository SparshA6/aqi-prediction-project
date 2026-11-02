"""Single place for project paths, so scripts work no matter where they're run from."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
CITIES_CSV = CONFIG_DIR / "cities.csv"
STATIONS_CSV = CONFIG_DIR / "stations.csv"

RAW_DIR = ROOT / "data" / "raw"
GEE_DIR = Path(os.environ.get("AQI_GEE_DIR", RAW_DIR / "gee"))
CPCB_DIR = RAW_DIR / "cpcb"
STATION_META_DIR = RAW_DIR / "station_metadata"
PROCESSED_DIR = ROOT / "data" / "processed"
ALIGNED_CSV = Path(os.environ.get("AQI_ALIGNED_CSV", PROCESSED_DIR / "aligned_dataset.csv"))

LOGS_DIR = ROOT / "logs"
