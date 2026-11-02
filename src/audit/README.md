# Alignment audit

Independent checks of the aligned dataset (recomputed from the raw files with separate code, not the
pipeline's own functions). Run from anywhere: `python src/audit/<script>.py`. Findings: docs/PROJECT_BRIEF.md, section 6g.

| Script | Checks |
|---|---|
| `1_spatial_and_identity.py` | station coordinates inside the city grid / not duplicated; each folder holds the right station's files; nearest-grid-cell match equals a brute-force haversine recompute |
| `2_independent_recompute.py` | recomputes AQI (UTC-day mean/max/valid hours) for EVERY aligned row straight from the xlsx files; completeness (no missing station-days); re-looks-up every satellite/weather value in the raw gee files |
| `3_hour_of_day_scan.py` | correlates the AQI at each hour-of-day with the weather day, to locate where the time alignment peaks |
| `4_daily_target_definitions.py` | compares candidate daily AQI definitions (mean of UTC bin, IST-day mean, single trailing-window readings) against same-day / adjacent-day weather and satellite NO2 |
| `5_dates_units_coherence.py` | Earth Engine date semantics (UTC), units/ranges, AQI sanity (cap, flat-lined sensors), per-station spatial coherence |

Scripts 2-5 read ~1300 xlsx files and take a few minutes each. Script 5 makes a few Earth Engine calls.
