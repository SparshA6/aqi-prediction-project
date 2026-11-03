# AQI Prediction — attention-faithfulness study



Multimodal spatiotemporal AQI prediction, and an audit of whether attention weights are a *faithful*

explanation of what drives the predictions (vs SHAP / LIME / perturbation / TCAV, across cities).

Full context, research questions and status: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md).

## 12 October follow-through

Paired 2026 error analysis and corrected city-wise attention summaries are in
`results/followthrough/oct12-v1/Results_Report.txt`. The dashboard includes
same-date GRU/weather/persistence comparisons and downloadable error and attention tables.

A 30% GRU / 70% archived-weather XGBoost blend is frozen in
`results/ensemble/runs/prospective-oct13-v1`. Its weight was chosen using October
2025 only, under sharp-change precision and recall constraints. It has **no fresh
test score** and does not replace the verified GRU default.

The prospective target period is **13 October-31 December 2026**. Ground AQI
exports use the same CAPTCHA-gated downloader as the historical years. On
2 November the portal listed October but had no November row, so expect a month
to appear only after it ends. Rerun in early December and early January, and
again for January after that month ends, because the 31 December target needs
the 05:00 IST reading on 1 January. Stations missing any requested month are
downloaded again and existing files overwritten.

```powershell
python src/cpcb_downloader.py --auto --year 2026 --months 10 11 12 --output-root data/raw/cpcb_prospective
python src/cpcb_downloader.py --auto --year 2027 --months 1 --output-root data/raw/cpcb_prospective
```

`src/prospective_tracker.py record` cannot currently produce before-target
records. It needs the previous day's AQI before target-day UTC midnight, which
monthly exports do not provide, and complete earlier-issued weather, which exists
only from 23:00 UTC before the target. `tracker status` reports
`target_has_prior_day_aqi` and `weather_failed_cities`. The evaluation below does
not depend on those records: it scores the frozen models out of time, from later
exports and archived forecasts.

After the period, prepare new inputs in separate directories:

```powershell
python src/prepare_prospective_inputs.py ground --raw-root data/raw/cpcb_prospective --output data/processed/prospective_ground_v1
python src/prepare_prospective_inputs.py weather --output data/processed/prospective_weather_v1
python src/models/ensemble_followthrough.py status
python src/models/ensemble_followthrough.py evaluate --aligned data/processed/prospective_ground_v1/aligned_dataset.csv --weather data/processed/prospective_weather_v1/daily.csv
```

Ground exports retain the configured station subfolder layout. The new weather
collector records the latest actual earlier issue time, requires a complete
24-hour forecast for each daily variable and keeps missing forecasts missing.
The evaluator opens on **1 January 2027 UTC**, checks frozen inference hashes
and input manifests, scores every model on the same rows and refuses to
overwrite its final test. It reports coverage exclusions and the predeclared
promotion criteria. Previously examined 2026 targets cannot validate this blend.



## Layout



| Path | What |

|---|---|

| `config/cities.csv` | cities to study: portal state/city names, satellite-grid bounding box, climatic zone |

| `config/stations.csv` | every CPCB station: portal name, city, lat/lon, `data_dir`, notes |

| `src/` | scripts (see below); `src/legacy/` = original single-station downloader |

| `data/raw/gee/<City>.csv` | satellite (NO2, HCHO, AOD) + ERA5-Land meteorology on the city grid, daily |

| `data/raw/cpcb/<City>/<Station>/` | 12 monthly hourly-AQI `.xlsx` exports per station |

| `data/raw/station_metadata/` | all-India station coordinates; the portal's own per-city station lists |

| `data/raw/legacy/` | the original hand-collected files |

| `data/processed/aligned_dataset.csv` | modelling-ready table, one row per station-day |

| `src/models/` | baseline models: feature table + splits, model grid, evaluation, results summary |

| `results/baselines/` | metrics (long format) and predictions for every model / feature set / protocol |

| `src/audit/` | independent alignment checks (see its README) |

| `docs/` | project brief, IEEE synopsis, literature review / novelty analysis |

| `logs/` | download and extraction logs |



## Pipeline



```bash

pip install -r requirements.txt

earthengine authenticate                        # once; Earth Engine project id is set in src/gee_extract.py



python src/build_config.py                      # only when adding a city (edit CITY_SPEC first)

python src/gee_extract.py [--city Lucknow ...]  # satellite + met  -> data/raw/gee/

python src/cpcb_downloader.py [--auto]          # station AQI      -> data/raw/cpcb/   (solve the CAPTCHA once)

python src/combine_with_aqi.py                  # join by nearest grid point + date -> data/processed/



python src/models/run_baselines.py --protocol chrono|loco|season_cv   # isolated experiment -> results/baselines/runs/

python src/models/summarize_baselines.py                             # summarizes the historical result files

```



Both downloaders are resumable (rerun to continue). All scripts find the project root themselves, so they

can be run from any directory.



## Things to know



- `date` means the **UTC** calendar day everywhere (ERA5-Land daily aggregates are UTC); CPCB's IST hours are

  shifted to UTC before aggregating to daily AQI.

- Cloud-masked satellite values (NO2, HCHO, AOD) are NaN; the ERA5 weather columns are always present.

  `satellite_bands_present` (0-3) in the aligned table says how many satellite values a row has.

- **Use `AQI_eod` as the target, not `AQI_mean`.** CPCB's hourly AQI is a trailing 24 h average, so the mean of the day's

  hourly values is blurred across two days; `AQI_eod` is the reading whose window matches the day (see docs/PROJECT_BRIEF.md 6g).

  `aqi_flat_day` marks stuck-sensor days.

- `climatic_zone` labels in `config/cities.csv` are our own qualitative labels, not a formal classification.





Sequence models (environment-only GRU/LSTM): `python src/models/run_sequence.py --protocol chrono|loco|season_cv` then `python src/models/summarize_sequence.py` (results in `results/sequence/`, tables in `docs/SEQUENCE_RESULTS.md`).



Attention model: `python src/models/run_sequence.py --protocol chrono --archs Uniform TA FA TA+FA --sets SW-w7 SWT-w7 --out attention [--save-models]`, then `python src/models/summarize_attention.py` (tables in `docs/ATTENTION_MODEL_RESULTS.md`).



## Reliable experiments



Each new runner invocation creates a separate directory under `results/<family>/runs/`.

`--quick` runs go under `results/<family>/smoke/`. Existing metrics and checkpoints are preserved.

Use `--run-name` to choose a readable name; an existing name is rejected. Each run records its

arguments, dependency versions, data and source fingerprints, expected folds, and completion status.

`requirements-tested.txt` records the tested direct dependency versions; it is not a complete environment lock.



```bash

python -m unittest discover -s tests -v

python src/models/check_results.py

python src/models/run_baselines.py --protocol chrono --models Ridge XGBoost --sets SW SW+G+T --run-name baseline-chrono-v2

python src/models/check_results.py --results-dir results/baselines/runs/baseline-chrono-v2

python src/models/summarize_baselines.py --results-dir results/baselines/runs/baseline-chrono-v2

```



The last command saves `summary.md` inside that run. Existing sequence and attention summary scripts

read the historical result folders by default. Use the metrics inside the printed run directory for new

experiments; do not silently copy partial or quick results into those historical folders.

`--folds loco:Mumbai` selects one fold for a diagnostic run. Such a run is not a full eight-city experiment.



Sequence and spatial training now reserve the latest 15% of training dates for early stopping,

purge the window overlap at the boundary, and fit input normalization only on the fitting rows.

This differs from the older random-week procedure: rerun complete comparisons with the new procedure

before drawing new accuracy conclusions. The outer chronological test is still reserved for final evaluation.



`AQI_band` now classifies `AQI_eod`; `AQI_mean_band` preserves a separate category for the daily mean.

The active dataset is `data/processed/aligned_dataset.csv`. The root-level CSV is an obsolete snapshot

without the current target and should not be used for modelling.



## Explanation audit



A first temporal-attention audit is available for the saved chronological TA and TA+FA checkpoints:



```bash

python src/models/audit_attention.py --model TA --set SWT-w7 --sample 300 --random-repeats 10

```



It writes row-level attention and leave-one-day-out sensitivity, deletion curves for attention,

random and reverse rankings, paired gaps with date-cluster bootstrap intervals, and city/regime summaries

to a new `results/faithfulness/<run>/` directory. It uses each checkpoint's saved preprocessing.

The replacement step is the mean transformed training step, and larger deletion changes indicate greater

sensitivity to removed inputs, not better prediction accuracy. Attention weights pool recurrent hidden

states, so a weight is not a direct attribution to that input day. Treat these results as exploratory;

repeat with other replacement methods, more rows, more seeds, and new validation-consistent checkpoints.



The historical spatial LOCO files are missing Mumbai. `check_results.py` reports the missing fold,

and `summarize_spatial.py` omits that incomplete comparison instead of reporting a seven-city mean as eight.

The spatial summary includes complete chronological and seasonal results and saved attention profiles.



## Research priorities



1. Finish matched experiments across all eight cities with the updated validation procedure. Choose

   hyperparameters and models using validation results, then evaluate the test set once.

2. Expand the explanation audit to feature groups and spatial cells. Add gradient/SHAP comparisons,

   multiple perturbation baselines, uniform controls, and retraining-based checks where feasible.

3. Report uncertainty and paired comparisons by city, season, satellite availability, and high-pollution

   regime. Keep forecast accuracy and explanation quality as separate outcomes.

4. Define the operational task: same-day daily satellite/weather inputs support retrospective estimation.

   For a next-day forecast, shift the target and use only inputs available when the forecast is issued.

5. Extend to multiple years and physically meaningful static features before adding model complexity.

   The present date ranges and splits are explicitly fixed to 2023 and must be generalized first.

6. Build a dashboard after the scientific evaluation is stable. Show the forecast horizon, prediction

   uncertainty, data availability and measured explanation limitations alongside the AQI estimate.



## Both prediction modes and the dashboard



The dashboard distinguishes retrospective same-day estimation from a historical next-day forecast.

The next-day pipeline uses observed AQI history and known calendar/location features; it excludes

same-day reanalysis and future satellite retrievals. Model selection uses September, interval calibration

uses October, and final scoring uses November/December. Persistence participates in selection.

The nominal 90% interval is evaluated empirically; temporal dependence and shift mean coverage is not guaranteed.



```bash

python src/models/run_forecast.py --run-name next-day-new

python -m pip install -r requirements-dashboard.txt

python src/dashboard/app.py

```



Open http://127.0.0.1:8765. Explore results lets you pick a model, city, station and day, then compare the model with the measurement on a zoomable chart, an accuracy table, a scatter plot, a station map and a sortable table. Click a chart day, press play, or drag to zoom. Try a forecast runs the frozen GRU for a city or from 28 readings you supply. How accurate? holds the test scores, the 2026 weather comparison, the attention results and the prospective status. Guide explains the terms. The address bar keeps the current selection, so a link or refresh reopens the same view.

It uses historical observations, not current AQI or real-time forecasts. The local service is bound to loopback.



## Full corrected study and expanded audits



```bash

python src/models/run_study.py --name corrected-v2 --seeds 3

python src/models/run_study.py --name corrected-v2 --seeds 3 --resume

python -m pip install --target .deps -r requirements-xai.txt

python src/models/audit_groups.py --kind feature --model FA --sample 100 --shap-samples 64

python src/models/audit_groups.py --kind spatial --model S-SA --sample 100 --shap-samples 64

```



The study runs all three protocols across baselines, sequences, attention and spatial models sequentially.

It validates each completed experiment and automatically audits new chronological attention/spatial checkpoints.

Status is saved in `results/studies/<name>/status.json`; failure stops the suite and keeps logs and prior results.

Resume rejects changed data/model code or seed counts. Repeated partial attempts get distinct result directories.

The dashboard selects the new same-day spatial model using validation results once that experiment is complete.



Feature audits group wind-direction sin/cos together and satellite measurements with their missing indicators.

Spatial audits preserve cell offsets and presence masks, and restrict sampled rows to full patches.

Both compare SHAP expected-gradient importance, attention, random and reverse rankings under training-mean

and same-city seasonal-donor replacements. These are sampled sensitivity tests, not causal explanations.



## Collecting and staging multiple years



```bash

python src/gee_extract.py --start-date 2024-01-01 --end-date 2024-12-31 --output-dir data/raw/gee_2024

python src/cpcb_downloader.py --auto --year 2024 --output-root data/raw/cpcb_2024

python src/prepare_multiyear.py --wait --resume

```



The CPCB portal may require a human CAPTCHA. Separate output folders protect the active 2023 experiment.

The staging script waits for complete city-year grids and station-month exports, checks grid consistency,

and writes a separate aligned table and GEE inputs in `data/processed/multiyear_2023_2024`.

The staged dataset requires a coverage/data-quality review before being used as a research benchmark.

The modelling calendar and sequence indices now span multiple years without December/January wrapping.

To use the staged data, set `AQI_ALIGNED_CSV` and `AQI_GEE_DIR` to those staged paths before starting a new study.

The chronological split reserves September/October of the last observed year and tests November/December.

Existing report titles may refer to 2023; update the scientific write-up to match the new dataset.


## Physical station covariates and practical forecast input

`src/extract_station_covariates.py` extracts SRTM elevation and WorldCover land-use fractions for station buffers. `src/models/run_static_ablation.py` compares ENV-T with ENV-T+P under all eight city holdouts. The completed `physical-features-loco-v1` run found no consistent improvement; retain the paired city uncertainty and individual fold scores in reports. Covariate provenance is stored with the data.

In the dashboard's next-day mode, enter an observed UTC day and AQI for the selected station, then choose **Estimate next day**. This uses the forecast chosen using validation data. The current selection is Persistence. Historical interval coverage is shown separately from nominal coverage. The standalone HTML snapshot displays historical results; custom forecasts require the local dashboard service.

## GPU training (NVIDIA CUDA)

The project-local `.venv-gpu` environment contains CUDA-enabled PyTorch 2.10.0 (CUDA 12.8). Activate it or call `.venv-gpu/Scripts/python.exe` directly. Sequence, attention and spatial runners accept `--device cuda`, `cpu` or `auto`; CUDA requests fail clearly if unavailable. Model weights are saved on CPU so the dashboard and explanation audits can still load them. Existing CPU results are retained with their original provenance.

```powershell
.\.venv-gpu\Scripts\python.exe src/models/run_study.py --name corrected-v2-gpu --seeds 3 --device cuda --resume
```

The GPU continuation carries forward completed CPU experiment jobs and restarts unfinished jobs in separate directories. Source snapshots and a migration record identify which jobs were inherited. Classical sklearn baselines continue on CPU; neural training uses CUDA. CPU and GPU floating-point results may differ.

## Untouched-year evaluation

The checked 2023–2024 dataset preserves the original 2023 rows, validates all eight city grids and 107 station-month collections, and records station coverage and raw-input hashes. Ingestion is resumable via verified station caches. The Bandra Kurla Complex IITM station has no usable 2024 daily targets; coverage is reported rather than fabricated.

```powershell
python -B src/prepare_multiyear.py --resume
.\.venv-gpu\Scripts\python.exe -B src/models/run_year_holdout.py --run-name another-unique-year-run --device cuda --seeds 3
```

The completed `2023-to-2024-v1` experiment fits January–September 2023, selects using October, calibrates with November–December, and tests all eligible 2024 days. Both same-day estimation and rolling next-day forecasting include persistence. Neural preprocessing fits only purged inner training rows; 2024 labels never fit or select a model. Outputs include city/season metrics, selected intervals, weekly paired bootstrap comparisons, seed-selection scores and provenance. The 2024 holdout has now been evaluated; subsequent model development requires a new untouched period for a fresh generalization claim.

## Next-day development with a fresh 2025 test

`run_forecast_development.py` trains on January 2023–September 2024, selects using October 2024, and calibrates using November–December. Candidates include persistence, short/28-day history, residual Ridge/RandomForest/XGBoost, delayed weather, and a three-seed 28-day GRU. XGBoost and GRU use CUDA; sklearn models use CPU. No 2025 table is loaded during development. Saved selection, model hashes and calibration are frozen before the test.

Weather uses only lag 7/14 ERA5-Land observations, allowing the usual five-day release delay plus a margin. Final retrospective reanalysis does not prove publication-vintage or Google ingestion availability. Treat weather candidates as exploratory until those times are verified. Past AQI is assumed received at issue time. Selection/calibration results are development scores; neither validates a live forecast.

```powershell
.\.venv-gpu\Scripts\python.exe src/models/run_forecast_development.py --run-name a-new-unique-name --device cuda --seeds 3
.\.venv-gpu\Scripts\python.exe src/gee_extract.py --start-date 2025-01-01 --end-date 2025-12-31 --output-dir data/raw/gee_2025
.\.venv-gpu\Scripts\python.exe src/cpcb_downloader.py --auto --year 2025 --output-root data/raw/cpcb_2025
.\.venv-gpu\Scripts\python.exe src/prepare_2025.py --resume
.\.venv-gpu\Scripts\python.exe src/models/evaluate_forecast_2025.py --development-run 2025-ready-v1 --run-name locked-2025-v1 --device cuda
```

The current `finish_forecast_2025.py` local pipeline waits for downloads and frozen training, validates 2025 data while preserving checked 2023–2024 history, runs the locked test, and exports results to the chat outputs folder. It does not refit or select using 2025. Missing station-month exports require a downloader retry; CAPTCHA remains human-only. Existing results cannot be overwritten. The dashboard remains on its validated benchmark until a fresh test is complete and reviewed.

The completed `locked-2025-v1` test evaluated 30,263 eligible days at 107 stations in eight cities. The three-seed GRU-28days selected in October 2024 achieved RMSE 32.516 versus persistence 35.388 (8.1% lower). Paired weekly RMSE gain was 2.872, with a 95% bootstrap interval [2.168, 3.605]. RMSE improved in all eight cities; Bengaluru MAE was slightly worse. Nominal 90% intervals covered 95.4% of this test. Overall scores weight station-days, with Delhi contributing 12,387 rows. Missing days and year-to-year changes limit interpretation.

`verify_forecast_2025.py` independently recomputes overall/city/season scores and interval coverage, checks saved model/calibration hashes, reproduces sampled predictions on CPU, and checks protected 2023–2024 rows. The dashboard shows verified 2025 next-day charts and results downloads. Custom GRU forecasts require the latest AQI plus the immediately preceding 27 daily values in chronological order. These supplied values are not verified live observations. The selected GRU uses AQI history and calendar/location inputs; it does not use the exploratory delayed-weather candidates. The 2025 test is now evaluated and frozen; any further model selection needs a new untouched period.

## City dashboard and weather-assisted development

The dashboard now compares all eight cities on a common chosen UTC date, shows available/expected station counts, ranks stations, filters chart dates, overlays yesterday's baseline, exports filtered CSV, and summarizes sharp-change misses. City predictions are equal-weight station averages, not official city AQI. City chart RMSE is calculated from paired daily city averages; station errors remain separate. Station interval bounds are not averaged into a purported calibrated city interval. Environment-only same-day estimation is shown separately from history-assisted estimation; its model was selected using October 2023, not 2024 test labels.

The next-day tab offers station inference from 28 supplied observations, local CSV import (`date,aqi`, exactly 28 consecutive UTC dates), and batch city inference from saved historical observations. Batch inference excludes all target-day and later readings before building inputs. Missing/stuck days exclude a station from batch inference. These inputs and outputs are historical or user supplied; no live CPCB integration is configured.

`forecast_diagnostics.py` audits the frozen 2025 results without changing them. At a 50-AQI change threshold, the existing GRU detected 164 of 3,975 sharp changes in the correct direction (4.1% recall). Its overall RMSE improvement does not establish reliable spike detection.

`extract_forecast_weather.py` caches public Open-Meteo Previous Runs GFS forecasts for eight city mean coordinates. Each target-day hour uses `previous_day1`, meaning a forecast made 24 hours earlier. Daily summaries require 24 available hourly values; no final reanalysis is substituted. City-scale weather and unverified original ingestion latency remain limitations.

`run_forecast_2026_development.py` now uses January 2024–September 2025 for fitting, October 2025 for selection, and November–December 2025 for interval calibration. The earlier 2025 benchmark remains frozen separately, and 2025 is explicitly development data for these new candidates. Four AQI-change XGBoost candidates train on CUDA and are compared with persistence and the frozen GRU on the same dates. The weather-assisted depth-5 model selected in October had RMSE 31.803, versus 36.228 for the frozen GRU and 37.262 for persistence. These are development scores, not fresh-test results. `verify_forecast_2026_development.py` recomputes all scores/calibration and reproduces sampled predictions on CPU.

The running `finish_forecast_2026.py` worker waits for all 963 January–September station-month files, rejects missing/duplicate months and conflicting hourly values, stages actual AQI with source hashes, and evaluates the frozen candidate once on 2026. The dashboard exposes the separate 2026 chart only after verification passes. It also displays download and test status. The worker stops after 24 hours if downloads are incomplete; rerun it to resume after resolving missing exports. CAPTCHA prompts require the human user.

```powershell
.\.venv-gpu\Scripts\python.exe src/cpcb_downloader.py --auto --year 2026 --months 1 2 3 4 5 6 7 8 9 --output-root data/raw/cpcb_2026
.\.venv-gpu\Scripts\python.exe src/finish_forecast_2026.py --outputs "C:\Users\agraw\Documents\Codex\2026-10-10\c-mystudymaterial-aqi-prediction\outputs"
```

The September 30 UTC end-of-day target requires October CPCB input and remains missing in this nine-month download. The model is never refitted using 2026 targets. Fresh performance and uncertainty must be assessed when the download/test finishes.

The completed 2026 comparison retried all 13 missing stations; each again returned no month exports. The available-data test contains 20,086 eligible station-days from 91 stations across all eight cities. Another three stations had monthly files but no usable daily targets. Availability and quality gaps are explicit; missing readings were never imputed into target labels. The comparison protocol and promotion criteria were sealed before scoring.

Weather-assisted RMSE was 30.270 versus 31.707 for the frozen GRU and 34.668 for persistence on the same days. The weather model improved RMSE in every city. Its paired weekly RMSE gain versus GRU was 1.437 (95% interval 0.376–2.614). Nominal 90% intervals covered 94.9%. Sharp-change recall increased from 6.7% to 18.2%, but precision fell from 57.5% to 52.8%. Because the predeclared promotion rule required no loss of either recall or precision, the GRU remains the dashboard default. The separate **Next-day · 2026 weather** tab exposes the verified candidate, city comparison, sharp-change scores and downloads. Further tuning now requires a new untouched test period.

`verify_forecast_2026.py` independently verifies actual targets, exact previous-calendar-day baselines, all 27 overall/city metric groups, intervals, model hashes and exclusions. The dashboard download/chart checks match the frozen outputs. The full test suite contains 23 passing checks.
