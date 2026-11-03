# AQI Explainability Research Project — Continuation Brief

*Handoff prompt — paste this whole thing to a teammate (or to Claude/any assistant) to get them fully up to speed.*

---

## 1. What this project actually is

We are building a **multimodal spatiotemporal AQI (Air Quality Index) prediction model**, but the predictor is *not* the contribution. The real research question is:

> **Does the attention mechanism inside the model give a faithful account of what actually drives its AQI predictions — or does it just look like an explanation?**

An attention weight is not automatically a predictive/causal contribution. Most "explainable AQI" papers show an attention heatmap or a SHAP plot and call it an explanation without ever testing whether it's *faithful* (i.e., whether it reflects what the model actually computed). We are explicitly treating "attention weight ≈ predictive importance" as a **hypothesis to test**, not a fact to assume — by comparing attention against independent explainability methods (SHAP, LIME, permutation/perturbation, and optionally TCAV) and against known atmospheric physics.

**Do not frame this as "we built an attention-based AQI model and explained it with SHAP."** That framing is already crowded in the literature (see Section 8). Frame it as: *an audit of whether attention's explanations can be trusted, and under what conditions.*

## 2. Scope decision: multi-city, not Delhi-only

Earlier drafts of this project were scoped to Delhi only. **That has been changed.** The design is now **multi-city**: the model is trained and evaluated across several cities spanning different climatic/emission regimes (e.g. a stagnation-prone plains city, a coastal city with strong ventilation, a semi-arid dust-influenced city), with an explicit **leave-one-city-out (cross-region transfer)** evaluation protocol. Cross-city consistency of explanations is now one of the core research questions and contributions — not an afterthought.

**Status (2026-09-20):** the multi-city dataset now exists -- 8 cities, 107 stations, 31,813 aligned station-days for 2023 (see 6f). Remaining caveats: coastal grid matching, Delhi = 39% of rows, seasonal satellite availability.

## 3. Research questions

**RQ0 (secondary, predictive):** Does each added data source / architectural component (satellite → +weather → +spatial → +temporal → +attention) actually improve AQI prediction, overall and during high-pollution episodes?

**RQ1:** Do high-attention features also have high importance under SHAP, LIME, and grouped permutation/perturbation?

**RQ2:** Does perturbing high-attention features change the prediction more than perturbing low-attention features (via comprehensiveness/sufficiency, deletion curves, remove-and-retrain)?

**RQ3:** Are attention rankings stable across AQI levels and across individual monitoring stations?

**RQ4:** Does attention remain faithful during extreme-pollution episodes, or does faithfulness degrade in the high-AQI tail?

**RQ5:** Where, and by how much, do attention and independent XAI methods disagree (quantified with feature-/rank-/sign-agreement metrics)?

**RQ6:** When methods disagree, which is more consistent with established atmospheric relationships (dispersion with wind speed, accumulation under stagnation, boundary-layer venting, hygroscopic growth with humidity)?

**RQ7 (multi-city specific):** Are attention rankings and their faithfulness consistent across cities and climatic regimes? Do explanations for a city *held out of training* stay faithful and physically plausible?

**RQ8 (concept-level, extension):** Is the model's internal representation sensitive to the physically defined concept of **atmospheric stagnation** (low wind + low boundary-layer height + low precipitation) during high-AQI predictions, via TCAV — and does that agree with feature-level attention/SHAP?

## 4. Dataset — what's needed vs. what exists

### Currently on disk (single region — Delhi only)
- `AQI_Prediction_Environmental_Data.csv` (~118k rows): satellite AOD, HCHO, NO₂ + ERA5 meteorology (2m temperature, 2m dewpoint, surface pressure, 10m wind u/v, total precipitation) + lat/lon, **daily**, gridded around Delhi (~28.3°N, 76.7°E), for 2023.
- `CPCB_Data/` + loose CPCB files: hourly station AQI for **Alipur, Delhi** only, 2023 (plus a stray 2017 file).
- `cpcb_downloader.py`: script used to pull CPCB station data.

### What the multi-city design needs (work-package one)
1. **Pick 6–12 cities** spanning distinct regimes — e.g. plains cities prone to winter stagnation, a coastal city with strong sea-breeze ventilation, a semi-arid dust-influenced city. (India is the natural default since CPCB + Sentinel-5P + ERA5 all have full coverage, but nothing forces it to stay India-only.)
2. **Satellite retrievals** per city: NO₂, HCHO, AOD at minimum (from Sentinel-5P/TROPOMI + MODIS, same pipeline as the existing Delhi extraction — reuse the same GEE/API method). CO, SO₂, O₃ are optional extras if pollutant-level attention analysis is wanted later.
3. **ERA5 meteorology** per city: 2m temperature, 2m dewpoint (→ derive relative humidity), surface pressure, 10m wind u/v (→ derive speed/direction), total precipitation, **plus `boundary_layer_height`** (not yet pulled — required for the stagnation concept in RQ8).
4. **Spatial:** lat/lon per grid cell (already the pattern used); elevation/land-use are optional, not required for the core RQs.
5. **Ground truth:** CPCB (or the relevant national agency) hourly station AQI for **every** city in the set, tagged with station id. Reuse/extend `cpcb_downloader.py`.
6. **Alignment:** satellite + met + station data joined to the same location + a shared daily time window across all cities (aggregate hourly station AQI to daily, document how gaps/mismatches are handled). Tag every row with: city, climatic zone, AQI band, season.
7. **Fixed common date window** across all cities (pick a shared multi-year range if possible — the current single year, 2023, is a reasonable starting point to replicate per city).

## 5. Model / methodology

**Architecture:** a spatiotemporal model that takes a window of past observations (satellite + met + spatial + AQI history) and applies **attention across environmental features and across time steps** before a regression head, with city/location passed in as context so one model serves all regions.

**Baselines to compare against:** Linear Regression, Random Forest, SVR, XGBoost/Gradient Boosting, MLP, LSTM/GRU, Transformer/Temporal Fusion Transformer, and an **attention-ablated version of the proposed model** (direct control).

**Ablation ladder:** satellite only → + weather → + spatial → + temporal history → + attention. Tests whether each component actually helps.

**Evaluation protocols (two regimes):**
- *Within-city*: temporally ordered train/val/test splits per city, no look-ahead leakage.
- *Cross-city*: leave-one-city-out — train on all-but-one city, evaluate prediction **and explanation faithfulness** on the held-out city.

**Explanation methods to extract per prediction:** attention (raw + attention-rollout/flow-corrected), SHAP, LIME, grouped permutation/perturbation importance (grouped because satellite/met inputs are physically correlated — naive single-feature perturbation leaks through correlates). Optionally: TCAV concept vectors for "atmospheric stagnation."

**Faithfulness metrics:** comprehensiveness & sufficiency (area-over-perturbation-curve), deletion/insertion curves, remove-and-retrain (ROAR). Cross-method agreement via feature-/rank-/sign-/signed-rank-agreement metrics (Krishna et al.'s disagreement-problem framework). All of this reported **stratified** by city, climatic zone, AQI band, and season — not just in aggregate.

**Physical adjudication:** when methods disagree, score each side against a checklist of known atmospheric relationships (wind speed ↔ dispersion, stagnation ↔ accumulation, boundary-layer height ↔ venting, humidity ↔ hygroscopic growth) rather than just reporting "they disagree."

**Prediction metrics:** MAE, MSE, RMSE, R², reported overall / per city / per AQI band / for identified high-pollution episodes, under both within-city and leave-one-city-out protocols.

## 6. Current repo contents

Reorganised 2026-09-19 (the folder used to be a flat pile of 40+ folders and loose files). Layout:

```
AQI Prediction/
├── README.md                     quick start
├── requirements.txt
├── config/
│   ├── cities.csv                which cities, their portal state/city names, satellite-grid bounding boxes, climatic zone
│   └── stations.csv              every CPCB station: id, portal name, city, lat/lon, data_dir, notes
├── src/
│   ├── paths.py                  single source of truth for paths (scripts run from anywhere)
│   ├── build_config.py           derives cities.csv + stations.csv rows from the station metadata
│   ├── gee_extract.py            Google Earth Engine: Sentinel-5P NO2/HCHO + MODIS AOD + ERA5-Land met -> data/raw/gee/<City>.csv
│   ├── cpcb_downloader.py        CPCB portal: hourly station AQI, all cities/stations, one CAPTCHA per run
│   ├── combine_with_aqi.py       nearest-grid-point + date join -> data/processed/aligned_dataset.csv
│   └── legacy/                   the original single-station CPCB downloader
├── data/
│   ├── raw/
│   │   ├── gee/<City>.csv        satellite + met on each city's grid, daily, 2023
│   │   ├── cpcb/<City>/<Station>/*.xlsx      12 monthly hourly-AQI exports per station
│   │   ├── station_metadata/     all-India station list (coordinates) + the portal's own per-city station lists
│   │   └── legacy/               the original hand-collected files (Delhi Jan-2023 grid, misc CPCB exports, archive.zip)
│   └── processed/aligned_dataset.csv   the modelling-ready table (one row per station-day)
├── docs/                         this brief, the IEEE synopsis, the literature review / novelty analysis
└── logs/                         download / extraction logs
```

Run order: `python src/build_config.py` (only when adding cities) -> `python src/gee_extract.py [--city ...]` -> `python src/cpcb_downloader.py [--auto] [--city ...]` -> `python src/combine_with_aqi.py`.

**Paths in the older subsections below (6a-6e) predate this reorganisation** -- `stations.csv` is now `config/stations.csv`, `gee_data/` is `data/raw/gee/`, `CPCB_Data*` folders are `data/raw/cpcb/Delhi/<Station>/`, `aligned_dataset.csv` is in `data/processed/`, and `cpcb_downloader_multi.py` became `src/cpcb_downloader.py`. The reorganised pipeline was regression-tested: it reproduces the previous Delhi output exactly (6,704 rows, all shared columns identical).

Baseline models exist (`src/models/`, see 6h). The sequence models (LSTM/GRU/Transformer), the attention model and all XAI code do not exist yet.

### 6a. The new data pipeline

- **`gee_extract.py`** — pulls Sentinel-5P NO2/HCHO, MODIS AOD, and ERA5-Land meteorology from Google Earth Engine, sampled on a grid over each city in its `CITIES` config dict, for a date range. Writes one CSV per city to `gee_data/`. Needs `pip install earthengine-api` + `earthengine authenticate` (one-time, needs a free Google Cloud project registered for Earth Engine) + your project id filled into `EE_PROJECT` at the top of the file. Run `python gee_extract.py --dry-run` first to sanity-check grid size and date count before burning Earth Engine quota on the real pull. **Boundary-layer height (needed for the RQ8 stagnation concept) is stubbed out but disabled by default** — GEE's ERA5 mirrors don't reliably expose it; verify against the current Earth Engine Data Catalog or pull it separately via the Copernicus Climate Data Store before enabling `INCLUDE_BOUNDARY_LAYER_HEIGHT`.
- **`combine_with_aqi.py`** — parses CPCB's `aqi_hourly_station_level_*.xlsx` monthly exports (the format `cpcb_downloader.py` produces), aggregates to daily AQI (mean + max, configurable), matches each station in `stations.csv` to its nearest satellite/met grid point via a haversine BallTree, and merges everything on date into `aligned_dataset.csv`.
- **`stations.csv`** — the join key between CPCB stations and cities/grids. Has 7 Delhi rows now (see 6c), coordinates verified against a real source, not guessed. As you pick the multi-city set (Section 4), add one row per station here, plus point `data_dir` at that station's downloaded CPCB folder.

### 6c. Six more Delhi stations queued (not yet downloaded)

To get spatial variety *within* Delhi (useful for RQ3's location-conditioned faithfulness even before the real multi-city pull), `stations.csv` now lists 7 stations spread across the city -- Alipur (N, already downloaded) plus Anand Vihar (E), R K Puram (S), Najafgarh (SW, semi-rural contrast), Mundka (W, industrial), ITO (central, traffic), Dwarka-Sector 8 (far SW). Coordinates came from a real Delhi CPCB/DPCC station-metadata source fetched directly (raw file, not an AI summary) -- worth noting since an earlier *summarized* fetch of that same source had briefly mislabeled a different station's (Bawana's) coordinates as Alipur's; fetching the raw CSV directly and cross-checking caught and avoided that.

**The other 6 stations' AQI is not downloaded yet.** `cpcb_downloader_multi.py` is ready to do it -- same CPCB CCR portal flow as `cpcb_downloader.py`, extended to loop all 6 stations in one browser session so the CAPTCHA only needs solving once, not six times. **This step needs a human:** CPCB gates the portal behind a manual CAPTCHA, and the script blocks on a terminal ENTER press after you solve it -- that can't be automated from here. To run it:
```
python cpcb_downloader_multi.py
```
Solve the CAPTCHA once when the browser opens, press ENTER in the terminal, and it downloads all 12 months for all 6 stations into their own `CPCB_Data_<Station>/` folders (already wired up in `stations.csv`'s `data_dir` column). One caveat: ITO is CPCB-operated rather than DPCC-operated -- if it doesn't appear in the portal's Station dropdown the way the others do, the script prints the actual available options so you can pick a substitute. Once downloaded, rerun `combine_with_aqi.py` to fold them into `aligned_dataset.csv`.
- **Already validated:** I ran `combine_with_aqi.py` for real against the existing January-2023 Delhi grid (reshaped into `gee_data/Delhi.csv`) and the real Alipur CPCB files already in `CPCB_Data/` (all 12 months, 8760 hourly rows). It produced 16 correctly-aligned rows (AQI 213–392, "Poor"/"Very Poor" bands — right for Delhi winter). That's `aligned_dataset.csv` as it stands now — **it's a proof that the pipeline works, not the real multi-city dataset.** Regenerate it once `gee_extract.py` has been run for the real city set and full date range.
- **Correctness audit done 2026-09-18** (see conversation for full detail): nearest-grid-point matching verified against an independent brute-force haversine check (exact match); hourly->daily AQI aggregation and the `MIN_VALID_HOURS` filter verified against known partial days; all 12 CPCB month files verified to parse with correct calendar boundaries and no duplicate/leaked dates. **One real bug was found and fixed:** CPCB reports in IST but `ECMWF/ERA5_LAND/DAILY_AGGR` bins on UTC calendar days, so "date" on the two sides was silently misaligned by up to 5.5 hours — `combine_with_aqi.py` now shifts CPCB hours to UTC before aggregating (`ALIGN_TO_UTC_CALENDAR_DAY`, on by default). Also confirmed: the aligned row count (16/24) is bottlenecked by satellite **pixel-level** cloud/coverage gaps at the specific matched grid cell (verified per-date grid coverage ranges from 750 to 6942 of 6942 possible points) — not by AQI completeness or a join bug.

### 6b. Earth Engine is live and proven end-to-end (2026-09-18)

- **Cloud project registered:** `aqi-prediction-505817` (already filled into `EE_PROJECT` in `gee_extract.py`). Authentication is cached locally (`~/.config/earthengine/credentials`) -- whoever runs this next either reuses that machine's credentials or runs their own one-time `ee.Authenticate()`.
- **Ran a real pull**, not a dry-run: full production grid spacing (0.03 degrees, 648 points) over Delhi for all of January 2023, straight from live Earth Engine (Sentinel-5P NO2/HCHO, MODIS AOD, ERA5-Land met). Took **54.5s for 31 dates (~1.8s/date)** -- useful for estimating the real multi-city, multi-month run time. Result: **24 valid dates, 117,971 rows** -- matching the original `AQI_Prediction_Environmental_Data.csv` almost exactly (same date list, same row count), strong evidence `gee_extract.py` faithfully reproduces whatever pipeline built the original dataset.
- This replaced the earlier placeholder `gee_data/Delhi.csv` (which had been reshaped from the old static CSV, not actually pulled). `combine_with_aqi.py` re-run against the genuinely live-pulled data reproduced the same 16 aligned rows -- full pipeline confirmed working end-to-end against real Earth Engine, not just replayed old data.
- **Not yet done:** the real full-scope pull (multi-city, full year, production grid) -- that's still Section 9, step 2. At ~1.8s/date/city for a 648-point grid, budget roughly `cities x days x 1.8s` plus request overhead for coarser/finer grids; scale accordingly once the real city set and grid spacing are decided.

### 6d. Full year pulled for Delhi; found and fixed a data-integrity bug (2026-09-18)

- **Ran the full 2023 pull for Delhi** (resumed past the already-done January dates): 308/365 days have at least some satellite coverage, 284 of those at the full 648-point grid. Took a few minutes, consistent with the ~1.8s/date estimate. July is almost empty (1 day) -- expected, monsoon cloud cover blocks both Sentinel-5P and MODIS almost completely that month, and it's also genuinely the cleanest-air month, so this isn't a pipeline problem.
- **Bug found and fixed:** `gee_extract.py`'s resume logic (`already_done_dates`) compared dates as raw strings. The very first placeholder `gee_data/Delhi.csv` (built early in this project by reshaping the old static CSV for a smoke test, on a different ~6900-point grid, in `M/D/YYYY` format) didn't match the `YYYY-MM-DD` strings the live extractor uses, so its dates were never recognized as "done" -- every later live pull appended fresh rows on top instead of the file ever getting cleanly replaced. End result: `gee_data/Delhi.csv` silently accumulated 117,971 stale rows from the wrong grid alongside the real data. **Fixed** by comparing parsed dates instead of raw strings (immune to format differences), with a warning if a file can't be parsed cleanly instead of silently treating it as empty. Stripped the stale rows out (123,983 genuine rows now remain, exactly 648 grid points, matches the production grid). Worth remembering if anyone re-derives `gee_data/*.csv` files by hand again -- match the real extractor's grid and date format, or start from a clean file.
- **`combine_with_aqi.py` re-run against the corrected, full-year data:** the aligned dataset grew from 121 rows (7 stations x January only) to **1270 rows** (7 stations x most of the year). No duplicate (station, date) rows. Station ordering by mean AQI stayed consistent with the January-only snapshot (Anand Vihar/Mundka highest, Najafgarh lowest), which is a good sign the earlier finding wasn't a small-sample fluke.

### 6e. All Delhi stations on the CPCB portal (2026-09-19)

- The CPCB portal offers **46** Delhi stations (list saved in `cpcb_portal_delhi_stations.txt`). `cpcb_downloader_multi.py` now reads `stations.csv`, skips stations that already have 12 months, tolerates per-station failures, and has an `--auto` flag (waits for the CAPTCHA modal to close instead of a terminal ENTER, so it can run detached; the CAPTCHA is still solved by a human in the browser window).
- **First run: 32 stations downloaded for all of 2023 (5,607 aligned rows, up from 1,270 with 7 stations).** A second run added 7 more -- see below.
- **Portal name != my source list's name for 7 sites:** the portal calls them `- IITM` where the coordinate source says `- IMD` (Aya Nagar, Burari Crossing, CRRI Mathura Road, IGI Airport T3, Lodhi Road (now `IMD Lodhi Road`), North Campus DU, Pusa). `stations.csv` now uses the portal names with the IMD coordinates -- **assumed** to be the same physical sites, not independently confirmed.
- **Six portal stations are deliberately NOT included:** JNU, IGNOU (Maidan Garhi), Talkatora Garden, Commonwealth Sports Complex, Delhi Cantonment, NSUT (Jaffarpur/West Campus). They were inaugurated 9 Feb 2026 (Zee News), so they have no 2023 data. Revisit if the study window moves to 2026. `New Moti Bagh - MHUA` was dropped too (not on the portal).
- **IITM-run stations have short 2023 records:** Chandni Chowk IITM has only 99 valid days and Lodhi Road IITM ~33 aligned rows. Consider dropping stations below a minimum-coverage threshold before modelling.
- **Second download run finished:** the 7 relabelled IITM sites all have data (135-199 aligned rows each). `IIT Delhi - IITM` downloaded 12 files but they are header-only -- no 2023 readings -- so it was dropped from `stations.csv` (its empty `CPCB_Data_IITDelhi_IITM/` folder can be deleted). **Final: 39 Delhi stations, 6,704 aligned rows** in `aligned_dataset.csv`; 33 grid cells, at most 2 stations per cell.

### 6f. Multi-city data pulled and aligned (2026-09-19/20)

- **Cities (8):** Delhi (IGP plains), Lucknow (IGP plains), Kolkata (humid delta), Mumbai (coastal west), Chennai (coastal southeast), Bengaluru (upland plateau), Hyderabad (semi-arid Deccan), Jaipur (semi-arid dust). Zone labels are our own qualitative assignment. Defined in `config/cities.csv`; grids cover every station of the city in the metadata (0.03 deg spacing, 110-648 points).
- **Satellite + met (`data/raw/gee/`):** all 8 cities pulled for 2023; no duplicate rows, no stray grid points.
- **AQI (`data/raw/cpcb/`):** 70 of 71 requested stations downloaded (Mumbai thinned from 30 to 14 by max spatial spread, so all cities are <=14 except Delhi's 39). 3 stations turned out to have no 2023 data (Nishant Ganj, Lucknow -- not on the portal; Manali Village, Chennai and BWSSB Kadabesanahalli, Bengaluru -- empty files) and were removed from `stations.csv`, with their empty folders parked in `data/raw/cpcb/_no_data/` (as was Delhi's IIT Delhi). Final: **107 stations**.
- **`data/processed/aligned_dataset.csv`: 31,813 rows** (Delhi 12,249 / Hyderabad 3,957 / Bengaluru 3,557 / Mumbai 3,417 / Kolkata 2,459 / Chennai 2,422 / Lucknow 2,099 / Jaipur 1,653). No duplicate (station, date). Weather columns are complete on every row; the satellite columns are NaN when cloud-masked -- use `satellite_bands_present` (0-3) to slice: 45% of rows have all three bands, 23% have none.
- **Former limitation 1 (monsoon gap) -- FIXED 2026-09-20.** The first pull dropped a grid point whenever *any* band was cloud-masked, which also discarded the always-available ERA5 weather (Jul-Sep had ~0-40 rows per city). `gee_extract.py` now unmasks the satellite bands (sentinel -9999 -> NaN), so a grid point/day is kept whenever ERA5 exists. Result: 14,376 -> 31,813 aligned rows; Jul-Sep rows 800 -> 7,506; every city has data in every month. Verified: all 14,376 earlier rows are present with identical AQI/satellite/weather values. The old masked pull is kept in `data/raw/gee/_v1_masked/`. **Side effect worth knowing:** the old table's pooled monsoon mean AQI was 125 (a biased sample of the few days that survived cloud masking); the corrected value is 80.
- **Remaining caveat on satellite availability:** it is still seasonal -- rows with >=1 satellite band are 76-98% of rows in most months but only 38-61% in Jul-Sep. So a *satellite-complete* subset under-represents the monsoon; weather-only or any-satellite subsets do not. Keep this in mind for the ablation ladder (weather-only step now has the full year) and for regime-conditioned faithfulness (RQ3/RQ4).
- **Known limitation 2 -- coastal stations match a farther grid cell.** Sea points have no land-only weather data, so a coastal station's nearest grid point with data can be far: median match distance is 1.1-1.7 km in most cities but **3.7 km in Chennai (max 7.7) and up to 9.0 km for Mumbai's Navy Nagar-Colaba**. `match_distance_km` is kept as a column; consider filtering (e.g. > 5 km) at modelling time. `MAX_MATCH_DISTANCE_KM` in `combine_with_aqi.py` is 15.
- **Imbalance:** Delhi is 39% of all rows (39 stations vs 6-14 elsewhere) -- weight or subsample for leave-one-city-out evaluation.
- **Thin stations (< 40 aligned rows):** Mumbai Ghatkopar-BMC (24), Sewri-BMC (25), Bengaluru Jigani (27), Kasturi Nagar (34), Delhi Lodhi Road-IITM (33).

### 6g. Alignment audit (2026-09-20) -- what was checked, what was found, what changed

Independent checks (scripts in `src/audit/`, recomputing from the raw files with separate code, not the pipeline's own functions):

**Passed -- no problem found**
- **Identity:** every station folder contains only that station's files; site IDs unique; no duplicate or near-duplicate (<150 m) coordinates; all stations inside their city's satellite grid.
- **Spatial match:** the matched grid cell equals a brute-force haversine nearest for every station (max distance discrepancy 0.00001 km).
- **Arithmetic:** all 31,813 aligned rows reproduce exactly from the raw xlsx files (AQI mean / max / valid hours: 0 mismatches), none are missing (0 valid station-days with grid data absent), and every satellite / weather value equals the raw grid file's value (0 mismatches over 10 columns).
- **File parsing:** `NA` cells (CPCB's missing marker) are read as missing, not zero; sparse day-rows are matched by their explicit day number, never by row position.
- **Earth Engine dates:** ERA5-Land daily images start at 00:00 UTC and every S5P / MODIS scene used falls inside that UTC day, so `date` = UTC day on the satellite/weather side, as assumed.
- **Units / ranges:** plausible everywhere (station pressure tracks altitude, dew point never exceeds temperature, AQI capped at 500). Tiny negative rainfall values (min -3e-8 m) are ERA5-Land numerical noise; negative NO2/HCHO are normal retrieval noise.
- **Station coordinates:** low within-city agreement at some Hyderabad/Bengaluru/Mumbai stations is NOT a coordinate problem -- those stations correlate weakly with *every* city-mate (max <= 0.42, no distance pattern), which points to noisy series rather than a misplaced station. Delhi/Kolkata/Lucknow stations agree strongly (median r 0.81-0.93).

**Found -- the daily AQI target was blurred over two days**
- CPCB's hourly AQI is a **trailing 24-hour running average** (CPCB, *How is AQI calculated?*: "AQI at 6am on a day will incorporate data from 6am on previous day to the current day"). The mean of 24 such values (`AQI_mean`) therefore has roughly half its weight on the *previous* day. Measured: `AQI_mean` correlates with **yesterday's** satellite NO2 (r=0.217) almost as strongly as today's (0.230); the naive IST-day mean is worse still (yesterday 0.238 > today 0.192), so the earlier UTC shift was right but not sufficient.
- **New column `AQI_eod`** = the hourly AQI at 05:00 IST on the next IST day, whose 24 h window covers (almost exactly) the UTC day the satellite/weather values describe. On the final table: same-day NO2 r=0.272 (vs 0.230 for `AQI_mean`), previous-day NO2 r=0.151 (vs 0.217), previous-day wind -0.062 (vs -0.126); lag-1 autocorrelation 0.58 (vs 0.66). Verified against an independent recompute (0 mismatches). It is available on 94.6% of rows (it needs that single hourly reading).
- **Recommendation: use `AQI_eod` as the primary prediction target** (fall back to `AQI_mean` only where `AQI_eod` is missing, and say so). Note that hourly AQI is itself a max-of-sub-indices on 24 h averages, so a daily target built this way is the CPCB-defined 24 h AQI ending at the day's close. `AQI_band` is still derived from `AQI_mean` (`DAILY_AGGREGATION` in `combine_with_aqi.py`) -- change it to `eod` if you switch targets.
- Caveat: all correlations are modest (|r| ~ 0.1-0.3) because daily 11 km reanalysis and once-a-day satellite columns are coarse proxies for local hourly pollution; this is what limits any model, not the alignment.

**Data-quality flags added**
- `aqi_flat_day` (174 rows): the daily AQI is constant all day and below the 500 cap -- a stuck sensor (14 further constant rows are legitimate saturation at 500). Concentrated at a few stations (e.g. Bengaluru BTM Layout 37 days, Peenya 19, Kasturi Nagar 18). Consider excluding flagged rows. `AQI_min` was added to support the flag.
- 17.5% of rows have fewer than 24 valid hours (2.9% fewer than 18); `AQI_valid_hours` is kept for filtering.

**Known remaining limitations (unchanged)**: the S5P NO2/HCHO value is a once-a-day (~13:30 local) snapshot compared with a 24 h AQI; ERA5-Land is 11 km; coastal stations match a farther land grid cell (see 6f).

### 6h. Baseline models (2026-09-20) -- first results

Code: `src/models/` (`data.py` feature table + splits + leakage self-check, `baselines.py`, `evaluate.py`, `run_baselines.py`, `summarize_baselines.py`). Full tables: `docs/BASELINE_RESULTS.md`; raw metrics/predictions: `results/baselines/`.

**Setup.** Target `AQI_eod`; stuck-sensor days dropped -> 29,923 station-days. Models: Ridge, RandomForest, XGBoost, SVR (RBF, fit on a fixed 6,000-row subsample -- O(n^2)), MLP; fixed *untuned* hyper-parameters, identical for every cell. Feature ladder: S (satellite) / W (weather) / SW / SW+G (+station lat, lon) / SW+G+T (+day-of-year, previous days' weather) / SW+G+T+H (+ the station's previous AQI, kept out of the main ladder because it dominates). References: persistence (yesterday's AQI) and a city-month climatology. Protocols: chronological (train Jan-Aug / val Sep-Oct / test Nov-Dec), leave-one-city-out (8 folds), season-block (4 folds of interleaved months). Scored on identical rows (`has_lag1` subset).

**Best test RMSE per ladder step (AQI points; lower is better)**

| step | chronological | leave-one-city-out (mean of 8) | season-block (mean of 4) |
|---|---|---|---|
| S satellite only | 119.7 | 72.4 | 80.3 |
| W weather only | 97.8 | 60.8 | 60.7 |
| SW | 93.5 | 58.5 | 57.2 |
| SW+G | 86.8 | 57.1 | 49.7 |
| SW+G+T | 72.1 | 57.2 | 49.0 |
| SW+G+T+H (AQI history) | 42.4 | 31.3 | 34.9 |
| *persistence (yesterday)* | 46.7 | 32.8 | 40.1 |
| *climatology* | 132.4 | 83.3 | 83.0 |

**What this says (be careful how the paper uses it)**
- **Environment-only models are modest.** In-city R2 is ~0.4-0.6 and they never beat persistence; only the AQI-history step does, and barely (42.4 vs 46.7). The XAI audit will therefore be explaining models that capture only part of the signal -- state this explicitly, and decide whether the audited model is environment-only (cleanest story for physical drivers) or includes AQI history (better accuracy, but history would dominate any attention/SHAP ranking).
- **Each ladder step helps in-city, but unevenly:** satellite adds ~4 RMSE on top of weather overall (more on days with all 3 bands present); +G (station coordinates) adds ~8 -- but coordinates let a model memorise station-level offsets, which is identification, not physics; +T adds little except under the chronological split, where the cyclical day-of-year lets a Jan-Aug model reuse winter behaviour on Nov-Dec.
- **Transfer to an unseen city is poor without history:** mean R2 across held-out cities is ~0 (trees, +0.06) or negative (Ridge/SVR/MLP); RMSE beats the climatology reference (~58 vs 83) but city-level offset dominates. Delhi is by far the hardest held-out city (RMSE 88-125) because its pollution regime differs most from the others. **Station coordinates and day-of-year *hurt* MLP/SVR badly in leave-one-city-out** (R2 -1.4 to -1.7: an unseen city's lat/lon is out of distribution) while helping trees only slightly -- so "+G" is not a transferable feature; consider physically meaningful static covariates (elevation, distance to coast, land use) instead.
- **Peaks are badly under-predicted** (regression to the mean): on Nov-Dec days with AQI >= 300 the best environment-only model is off by 60-100 points on average (bias -61 to -130 depending on step), and on "Severe" days by ~100. Relevant to RQ4 (faithfulness in extremes) -- explanations will be of a model that systematically misses them.
- **Caveats:** hyper-parameters untuned; single year (2023); leave-one-city-out shares dates across cities (weather is correlated across them); Delhi is 39% of rows; the chronological test is a distribution-shift test (train months are cleaner than test months).

## 7. Existing deliverables (read these first)

1. **`AQI_Synopsis_IEEE.pdf`** — the current IEEE-format synopsis: title ("Auditing the Faithfulness and Cross-City Consistency of Attention in Multimodal Spatiotemporal Air Quality Index Prediction"), abstract, all 4 authors, the full RQ list, methodology sketch, expected contributions, evaluation plan, scope/limitations, references.
2. **`Prior_Work_Review_and_Novelty_Analysis.pdf`** — a literature review covering: the attention-as-explanation debate (Jain & Wallace, Wiegreffe & Pinter, Serrano & Smith, Abnar & Zuidema, Bibal et al.), faithfulness-evaluation methodology (ERASER, ROAR, the disagreement problem), explainability in air-quality prediction, spatiotemporal multimodal AQI/PM2.5 models (including the closest prior work, arXiv:2407.11283, which our predictor architecture resembles), and TCAV/concept-based explanation + the atmospheric stagnation index. It ends with a novelty assessment (what's defensible to claim vs. what's already been done) and concrete search queries to verify novelty independently. **Read this before writing any "no one has done X" claim in the paper.**

## 8. Novelty guardrails — do NOT claim these as new

- A multimodal spatiotemporal AQI/PM2.5 predictor combining satellite + met + ground data (extensive prior work; a near-identical attention+LSTM+multitask architecture already exists — arXiv:2407.11283).
- Applying SHAP/LIME to an AQI model (common).
- Putting attention in an AQI model (common).
- "Attention is not automatically a faithful explanation" as a general claim (settled since 2019 in NLP; we're testing *whether/when* it holds here, not discovering it).

**What is defensible (conditional on independently verifying via the search queries in the review doc):** a rigorous faithfulness audit of attention specifically in the multimodal spatiotemporal AQI *regression* setting; physical-plausibility adjudication of XAI disagreement; regime- and **cross-city**-conditioned faithfulness, including transfer to a held-out city; TCAV on a physically grounded "stagnation" concept; and folding explanation quality into the ablation study.

## 9. Immediate next steps, roughly in order

1. **Pick the city set** (6–12 cities, spanning regimes) and the shared date window.
2. **Run the data pipeline for real**: fill in `CITIES` and `EE_PROJECT` in `gee_extract.py`, run it per city (dry-run first), download+add CPCB stations for each city into `stations.csv`, then run `combine_with_aqi.py` to produce the real `aligned_dataset.csv`. (The pipeline code exists and is smoke-tested — see Section 6a — this step is "run it at scale," not "write it.")
3. **Extend as needed**: verify/enable `boundary_layer_height` in `gee_extract.py` (required for RQ8's stagnation concept); add within-city and leave-one-city-out train/val/test splits on top of `aligned_dataset.csv`.
4. **Implement baselines + the attention model + the ablation ladder**; get RMSE/MAE/R² stratified as described.
5. **Implement the XAI extraction + faithfulness pipeline**: attention (raw + rollout), SHAP, LIME, grouped permutation; comprehensiveness/sufficiency/deletion/ROAR; agreement metrics.
6. **Physical-plausibility checklist** + disagreement adjudication.
7. **(Extension) TCAV** for the stagnation concept once boundary-layer-height data is in.
8. **Run your own systematic literature search** using the query strings in `Prior_Work_Review_and_Novelty_Analysis.pdf` (§12) before finalizing novelty claims in the write-up.
9. Write the full paper from real results.

## 10. Team

Saanvi Garg (240905702, Roll 68) · Sparsh Agrawal (240905708, Roll 70) · Dhruv Chamria (240905706, Roll 69) · Neelaksha Sisodiya (240905642, Roll 65) — Section CSE-C, School of Computer Engineering, Manipal Institute of Technology, MAHE, Manipal.


### 6i. Sequence models: environment-only GRU / LSTM (2026-09-20)

**Superseded numbers.** 6i and 6j report the first runs, which stopped early on randomly held-out weeks. The corrected runs use purged date-based validation (`results/*/runs/corrected-v2*`) and differ materially; use 6k for attention comparisons. `docs/SEQUENCE_RESULTS.md`, `docs/ATTENTION_MODEL_RESULTS.md` and `docs/SPATIAL_MODEL_RESULTS.md` also show the first runs.

Code: `src/models/seq_data.py` (windows, checked against the baseline table's lag columns), `seq_models.py`, `run_sequence.py`, `summarize_sequence.py`; results in `results/sequence/`, tables in `docs/SEQUENCE_RESULTS.md`.
Setup: input = per-day satellite (+missing indicators) + weather over a 7-day window (today + 6 previous days), NO AQI history. 1-layer GRU/LSTM (64 units) + small MLP head, MSE loss, Adam, early stopping on held-out whole weeks of the training rows, mean of 3 seeds, fixed untuned hyper-parameters. Input sets: `SW-w1` (today only, control), `SW-w7`, `SWT-w7` (+ day-of-year). Same three protocols, same scored rows as the baselines.

Test RMSE (best seq model per set vs best tabular env-only model):

| input | chrono | LOCO mean | season-CV mean |
|---|---|---|---|
| SW-w1 (GRU) | 88.8 | 58.3 | 54.0 |
| SW-w7 (best of GRU/LSTM) | 82.7 | 55.1 | 51.7 |
| SWT-w7 (best) | 77.9 | 54.1 | 50.6 |
| best tabular SW / SW+G / SW+G+T | 93.5 / 86.8 / 72.1 | 58.5 / 57.1 / 57.2 | 57.2 / 49.7 / 49.0 |
| persistence | 46.7 | 32.8 | 40.1 |

Findings: (1) the 7-day window helps by about 4-6 RMSE over the same-day-only control in every protocol, so temporal context carries real signal; (2) on the same-day-plus-window inputs the sequence models beat tabular models on SW (93.5 -> 82.7 chrono) and are best on LOCO, but they do NOT beat tabular models that have lat/lon + hand-built lags (SW+G+T) on chrono (77.9 vs 72.1) or season-CV (50.6 vs 49.0); (3) GRU vs LSTM are within noise; (4) no environment-only model reaches persistence anywhere; Delhi (87-89) and Lucknow (86-91, vs 61 for tabular) transfer worst; (5) peak under-prediction is WORSE than the tabular MLP (AQI>=300 bias -98 to -128 vs -61). Caveat: untuned, single year, 3 seeds. Implication for the XAI stage: the environment-only sequence model is a reasonable but modest model of AQI; explanations must be reported with its accuracy, and the attention variant should be compared against this GRU (same inputs, no attention) as its control.


### 6j. Attention model (environment-only) -- first results (2026-09-20)

Code: `AttnNet` in `src/models/seq_models.py` (run via `run_sequence.py --archs Uniform TA FA TA+FA --out attention`), `summarize_attention.py`; tables in `docs/ATTENTION_MODEL_RESULTS.md`; chrono models (3 seeds) saved in `results/attention/models/` for the XAI stage.
Variants (all: GRU-64 encoder, same inputs as 6i, 3 seeds, untuned): `Uniform` = attention-ablated control (fixed 1/7 weights over days); `TA` = learned temporal attention over the 7 day-states; `FA` = learned softmax gate over the input features of each day (inputs rescaled by F*beta); `TA+FA`. Plain last-hidden GRU (6i) is the second control.

Accuracy (test RMSE, chrono / LOCO / season-CV): SW-w7: GRU 84.4/55.4/51.7, Uniform 89.6/56.3/53.7, TA 84.1/55.2/52.8, FA 85.2/59.4/53.9, TA+FA 91.0/58.5/54.2. SWT-w7: GRU 77.9/54.2/50.6, Uniform 75.3/55.0/52.2, TA 77.3/54.0/51.6, FA 82.4/58.4/53.4, TA+FA 85.9/56.1/52.2.
=> Learned attention does NOT improve prediction: TA ~ plain GRU, and beats the uniform control by ~1-5 RMSE only on some settings (on SWT-w7 chrono the uniform control is best); feature gating hurts (worse LOCO by ~4). Differences of a few RMSE are within plausible seed noise (3 seeds, no CIs yet) -- do not over-read.

What the weights look like (descriptive only, NOT yet tested for faithfulness): temporal attention is concentrated on the target day and the day before (d-0 ~0.29-0.38, d-1 ~0.20, d-5/d-4 lowest, small bump at d-6), normalised entropy 0.65-0.78, mean profile very stable across seeds (r 0.90-0.99) but the weight varies a lot row to row (std ~0.2). Feature attention is close to uniform (entropy 0.95-0.97): temp_c is highest (1.4-1.9x uniform; likely a season proxy), then NO2, pressure, wind direction, wspd; rh and the missing-indicators lowest -- although humidity is physically expected to matter; seed-to-seed agreement of the feature profile is only r 0.64-0.86.
Implications for RQ: (1) the attention-weight profiles are candidate "explanations" to be compared with SHAP / permutation / leave-one-out / LIME / TCAV; (2) the feature gate is computed FROM the inputs, so its weights are not automatically an importance measure -- a good test of the RQ; (3) since the Uniform/GRU controls perform about as well, the honest expectation is that attention will turn out to be at best a partial explanation; do not pre-write that conclusion.


### 6k. Spatiotemporal neighbouring-cell patch model (2026-09-20ish)

Code: `src/models/spatial_data.py` (patch builder, checked against the single-cell windows: centre cell == 6i/6j inputs exactly), `seq_models.SpatialNet`, `run_spatial.py`, `summarize_spatial.py`; tables in `docs/SPATIAL_MODEL_RESULTS.md`; chrono models saved in `results/spatial/models/` for the XAI stage.
Setup: 3x3 patch of grid cells around the station (stride 3, ~10 km spacing -- stride 1 would just duplicate the native 0.03 deg grid against ERA5-Land's coarser ~0.1 deg resolution), each cell its own 7-day window, one GRU shared across cells, two-level attention (temporal pooling per cell, then spatial pooling across the patch, masked for absent/coastal cells). Cells are given only a relative offset (not lat/lon), so the model can attend to *where* a cell is without memorising which station it is. Variants: `S-Uniform` (attention-ablated control, uniform over days and cells), `S-TA`, `S-SA`, `S-TA+SA`.

Corrected test RMSE (chrono / LOCO city-macro / season-CV fold mean), from `results/attention_stability/gains.csv` (`src/models/report_attention_stability.py`, saved predictions, no retraining):

| input | Uniform | TA | FA | TA+FA | S-Uniform | S-TA | S-SA | S-TA+SA |
|---|---|---|---|---|---|---|---|---|
| SW-w7 | 84.16 / 55.89 / 57.06 | 84.90 / 54.16 / 56.61 | 80.56 / 58.16 / 57.48 | 91.43 / 56.42 / 59.39 | 80.17 / 57.07 / 56.69 | 81.48 / 53.93 / 56.16 | 83.76 / 56.19 / 53.78 | 82.86 / 54.07 / 54.69 |
| SWT-w7 | 78.36 / 55.11 / 54.74 | 84.39 / 55.23 / 54.71 | 82.28 / 60.27 / 56.29 | 94.53 / 57.96 / 56.32 | 75.33 / 57.19 / 54.54 | 83.64 / 56.34 / 55.43 | 65.84 / 56.79 / 53.07 | 75.97 / 54.16 / 53.66 |

Findings: whether attention helps depends on the configuration. S-SA gains 9.49 (95% interval 3.64 to 15.46) on SWT-w7 chrono but loses 3.58 (-5.54 to -1.61) on SW-w7 chrono. FA loses on SWT-w7 chrono and gains 3.59 (1.64 to 5.33) on SW-w7 chrono. The most consistent result is S-TA+SA in LOCO: 3.03 (2.08 to 3.95) on SWT-w7 and 3.00 (2.20 to 3.77) on SW-w7, with no chronological gain on either set. Intervals are paired calendar-week block bootstraps. They cover test-week sampling only, not training seeds, and are unadjusted across 36 comparisons. Weight maps in `docs/SPATIAL_MODEL_RESULTS.md` come from the first runs and need regenerating before they are cited.


### 6l. Year-holdout and next-day forecast evaluations (2026-10)

Separate from the main faithfulness study, these ask "how good is the frozen pipeline at forecasting, not just same-day estimation" using strict train/test year splits (`results/year_holdout/`, `results/forecast_2025/`, `results/forecast_2026/`; summarised in `docs/AQI_Prediction_Full_Project_Report.docx`, independently re-derived against the raw CSVs and confirmed to match).

- **2023 -> 2024 year-holdout** (same-day estimation, tabular Ridge with AQI-history lags): test MAE 23.00, RMSE 32.81, R2 0.892 (n=28,803).
- **2025 next-day forecast** (horizon = 1 day, excludes target-day observed AQI/satellite/weather; GRU, 28-day window): test RMSE 32.52, MAE 21.86, R2 0.887 (n=30,263).
- **2026 next-day forecast, weather-aware candidate** (XGBoost residual model, tree depth 5, 28 AQI lags plus Open-Meteo Previous Runs weather issued 24 hours before each valid hour): test RMSE 30.27, MAE 19.35, R2 0.848 (n=20,086); block-bootstrap (2,000 calendar-week draws) used for uncertainty.
- **Abrupt-change (>=50 AQI) detection, "all" row**: the weather-aware XGBoost reaches 18.23% recall / 52.82% precision vs. 6.65% recall / 57.46% precision for the frozen GRU -- better recall at a real precision cost, not a clean win.
- **Severe-AQI (>=300) and abrupt-change alert classifiers** (`results/event_alerts/`): fit on October 2024 GRU predictions, thresholds fixed on Nov-Dec 2024, evaluated once on 2025 only. Neither candidate passed its calibration promotion check, and neither was **deployed**: on severe AQI the existing rule had 82.9% recall / 83.7% precision vs. the candidate's 64.9% / 88.4%; on abrupt change the existing rule had 4.3% recall vs. the candidate's 0.2%. The frozen GRU default is unchanged.

Takeaway: compare models only within a cohort, because these three tests use different periods and rows. Each candidate beat persistence on its own rows. In 2026 the weather model also beat the frozen GRU on RMSE (gain 1.44, 95% interval 0.38 to 2.61), but it lost sharp-change precision, so the GRU stays the default.


### 6m. Corrected leave-one-city-out (LOCO) cross-city transfer (2026-10-21)

8-fold LOCO, summarised with **equal weight per held-out city** (city-macro RMSE, not row-weighted -- avoids letting Delhi's row count dominate). Source: `results/city_transfer/`, `docs/Followthrough_2026-10-21.md`.
SWT-w7: spatial **S-TA+SA 54.16** city-macro RMSE vs. spatial **Uniform 57.19** -- improves 5 of 8 held-out cities. Sequence **TA 55.23** vs. sequence **Uniform 55.11** -- no improvement on average.
These figures are SWT-w7 only. On SW-w7, sequence TA does gain (1.73, 95% interval 0.86 to 2.55, 5 of 8 cities), so 6k should be read alongside them. This is cross-city accuracy, not evidence about faithfulness.


### 6n. Prospective (forward-looking) evaluation pipeline (2026-10 onward)

Two separate mechanisms cover the window **13 October - 31 December 2026**:
- **Before-target ledger** (`src/prospective_tracker.py record`): writes an immutable forecast only before the target's UTC midnight. It has 0 records and cannot gain any through the current route. It needs the previous day's AQI, but the latest usable AQI on 2026-11-02 was 30 October and the portal had no November row. It also needs all 24 earlier-issued weather hours, available only from 23:00 UTC before the target. Records would need a same-day AQI source, which is not configured.
- **Frozen out-of-time evaluation** (`src/models/ensemble_followthrough.py evaluate`, opens 1 January 2027): scores the frozen 30% GRU / 70% weather XGBoost blend on later exports and archived forecasts. It does not read the ledger. It tests a frozen model on later data, not forecasts issued live. Archived forecast latency remains unverified.
- **Ground status, 2026-11-02**: 104/107 stations have October exports, with 2,581 usable station-days, of which 1,559 fall in the test window. Vasai West, Vile Parle West and Chhatrapati Shivaji Intl. Airport T2 were downloaded in earlier runs but were absent from the portal's Mumbai list on 2 November; the reason is unresolved, so retry them. 13 stations are excluded from the January-September 2026 test as having no exports, yet 12 of them have October 2026 exports, so those gaps may be download failures rather than missing data. Excluding them was not based on scores.
- **Weather provenance**: each snapshot saves `retrieval_audit.json` and `single_run_audit.json`. The 2026-11-02 snapshot failed for all 8 cities (read timeouts and HTTP 429), which `status` now reports as `weather_failed_cities`.


### 6o. Dashboard (`src/dashboard/`)

Flask app with a public static demo distribution (`share/`). Four tabs: **Explore results** (one selected station or city average and one selected day drive the answer panel, chart, accuracy table, scatter, station map and station table), **Try a forecast** (frozen GRU; city dates limited to 2025 because earlier dates are its training data), **How accurate?** (each model against yesterday's reading, 2026 weather comparison, attention across input sets, faithfulness audit, and the unscored October-December 2026 status), and **Guide**. It keeps three things separate that are easy to conflate: same-day estimates, historical next-day backtests (6l), and the prospective evaluation (6n, currently unscored). On opening, the day is the latest available; changing city or station keeps the day if that selection has a reading, otherwise it moves to the latest. The page loads `/api/dashboard`, which is `state()` with rows stored by column; `--export` embeds the same payload. Scores on the page are recomputed from those rows and match the saved headline RMSEs.
