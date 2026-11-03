# AQI Prediction - full local demo

This is the shareable demonstration build. It contains saved historical results, the frozen 2025 GRU checkpoint, and the minimum station AQI history needed for the station and city prediction forms. Raw CPCB spreadsheets, research training runs, credentials, logs, and the 4 GB development environment are excluded.

## Run

Use Python 3.13. On Windows, extract the ZIP and run `run_windows.bat`. On macOS or Linux, run `sh run_unix.sh`. The first run installs the CPU-capable dependencies in a local virtual environment. Then open <http://127.0.0.1:8765/>. Stop the server with Ctrl+C.

The server binds to `127.0.0.1`; it is intended for a local demo. Installation requires internet access to download packages, including PyTorch. The project does not need Earth Engine credentials or the CPCB downloader to run this build.

## What to demonstrate

- Same-day and environment-only retrospective estimates: 2024 held-out data.
- GRU next-day forecasts: 2025 held-out data and a custom station forecast from 28 consecutive supplied daily AQI values.
- Weather-assisted next-day forecasts: January-September 2026 saved paired comparison with GRU and yesterday's AQI.
- City forecast: choose a 2025 observation date with 28 valid earlier days; the result averages eligible station forecasts, not official city AQI. Earlier dates are not offered because they are the model's training data.
- Explore results: click, play or zoom the chart, compare each model with yesterday's reading, use the station map and table, and export CSV.
- How accurate?: test scores, the 2026 weather comparison, attention results across test designs, and the faithfulness audit.

The chart and model comparison are historical backtests. The station input form uses values provided by the presenter; it is not a verified live observation. The October-December 2026 prospective study has no score in this package. The 2023 data were used for training, selection, and attention research; they are not shown as an independent dashboard test year.

## Data and provenance

Ground-station AQI comes from CPCB exports processed to UTC daily end-of-day targets. The 2024, 2025, and 2026 rows are copied from the project's verified saved result runs. The model manifests and verification records remain beside each result. The included 2023-2025 history file contains only date, station ID, AQI end-of-day, and quality flag columns needed for the historical city prediction form. The frozen three-seed GRU files and their SHA-256 hashes are included in the 2025 development manifest. Saved attention, faithfulness and prospective-status summaries are included for the How accurate? tab; the prospective status is fixed at the time this package was built.

See `docs/AQI_Methodology_Submission.pdf` and `docs/AQI_Results_and_Discussion_Submission.pdf` for methodology and results. No training is performed by this demo package.
