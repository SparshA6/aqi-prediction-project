# AQI project follow-through — 21 October 2026

The public dashboard remains a demonstration of saved historical data. The prospective October–December 2026 evaluation is still in progress and has no valid score.

## Prospective collection

At the last status check, the CPCB October collection had **0/107 station exports**, **0 usable new station-days**, and **0 immutable before-target forecast records**. The existing downloader reported that October rows were unavailable. No target is backdated or scored. Continue to check for newly available CPCB files, quality-check them with `src/prospective_tracker.py ground`, and use `record --target YYYY-MM-DD` only before the target UTC midnight with complete earlier-issued weather and eligible prior-day AQI. Do not retry all 107 stations while the same unavailable-month condition persists.

## Weather provenance

New weather snapshots save `retrieval_audit.json` with per-city request/response times, HTTP Date when provided, URL, and raw-response SHA-256. A separate `single_run_audit.json` requests the named GFS 12 UTC run from the previous day when the 24-hour-lead weather is complete and that run is at least six hours past initialization. The response is saved and hashed. A failed exact-run audit does not make a weather input complete or alter the model input. The recorder verifies both audit hashes when present. Older snapshots remain readable.

Open-Meteo defines `_previous_day1` as values predicted 24 hours before valid time. The named [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) identifies a model's **initialization time**, which is different from public availability; it also does not prove that every `_previous_day1` value came from that run. The [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) does not itself identify a single run for the full daily sequence. The audit therefore records both sources without claiming exact identity or publication certification.

## Severe-AQI and abrupt-change alerts

The separate development experiment fits alert classifiers on October 2024 GRU predictions and fixes thresholds using November–December 2024. It evaluates the resulting rule on 2025 only; no 2026 target enters the experiment. On 2025 severe AQI (at least 300), the existing forecast threshold had **82.9% recall** and **83.7% precision**; the candidate had **64.9% recall** and **88.4% precision**. On abrupt absolute changes of at least 50 AQI, the existing rule had **4.3% recall**, while the candidate had **0.2% recall**. Both candidates fail the calibration improvement check, so neither is deployed. The point forecast and frozen GRU default are unchanged. See `results/event_alerts/` for source hashes, metrics, saved research candidates, and details.

## Cross-city evaluation

The corrected eight-fold leave-one-city-out runs were summarized with equal weight per held-out city. For SWT-w7, the spatial S-TA+SA model reached **54.16 city-macro RMSE**, compared with **57.19** for spatial Uniform, improving **5/8** cities. Sequence TA reached **55.23**, compared with **55.11** for sequence Uniform; the eight-city average did not improve. These are historical spatial-transfer results and do not establish attention faithfulness on held-out-city checkpoints. See `results/city_transfer/` for each fold, all model summaries, and source hashes.

## Demo

The dashboard now distinguishes same-day estimates from historical next-day backtests and from the user-input prediction demo. The comparison table explicitly says that its 2026 scores are historical, not prospective. On opening or changing a station/city, the comparison date follows that selection's latest available day, so the summary and station table agree. The homepage stays focused on prediction controls, city/station results, graph, and a concise model comparison; collection and research details remain in files and the API.
