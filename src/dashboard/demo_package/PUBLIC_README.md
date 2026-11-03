# AQI Prediction - public historical dashboard

**Live demo:** https://sparsha6.github.io/aqi-prediction-project-demo/

**Local prediction app:** Download `AQI_Local_Demo.zip` from the Guide tab of the live demo or from this repository.

Open `index.html` in a browser or place this directory on a static web host. It needs no server, account, or API key. It embeds saved 2024 same-day, 2025 GRU next-day, and January-September 2026 weather-model backtests.

Visitors can choose a model, city, station and day, click, play or zoom the chart, compare each model with yesterday's reading, search and sort stations, and export selected rows. The How accurate? tab holds the test scores, the 2026 weather comparison, and the attention and faithfulness results. The embedded values are fixed at the time this package was built. They are not live AQI readings or a score for the reserved October-December 2026 prospective study.

Ground observations are derived from CPCB station exports and processed into UTC daily targets. The methodology and results PDFs are included for context and limitations. The 2024, 2025, and 2026 comparisons have different test cohorts; do not rank their scores as one matched test.

Custom station and city forecasts require Python and the frozen model. Download `AQI_Local_Demo.zip` from the Guide tab for those functions. City chart values are averages of available station observations or predictions, not official city AQI.

For static hosting, upload `index.html`, `AQI_Local_Demo.zip`, and both included PDFs together. Do not upload development data, the raw CPCB files, credentials, or the training environment.
