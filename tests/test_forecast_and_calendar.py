"""Prevent next-day target leakage and multi-year window wrapping."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))

import numpy as np
import pandas as pd
from run_forecast import FEATURES, forecast_table
import seq_data


def station_rows(station, base):
    dates = pd.to_datetime(["2023-12-29", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-03"])
    return pd.DataFrame(dict(date=dates, station_id=station, city="Delhi", station_name=station,
                             station_lat=28., station_lon=77., aqi_flat_day=False,
                             AQI_eod=np.arange(len(dates)) + base))


class ForecastCalendarTests(unittest.TestCase):
    def test_target_is_next_calendar_day_same_station(self):
        a = pd.concat([station_rows("a", 10), station_rows("b", 100)])
        df = forecast_table(a)
        self.assertEqual(len(df), 6)
        self.assertTrue((df.date - df.feature_date == pd.Timedelta(days=1)).all())
        np.testing.assert_equal(df.y.to_numpy() - df.latest_aqi.to_numpy(), 1)
        self.assertFalse((df.feature_date == "2024-01-01").any())
        self.assertTrue(all(name not in FEATURES for name in ["y", "AQI_eod", "temp_c", "NO2"]))

    def test_targets_do_not_enter_features(self):
        a = station_rows("a", 10)
        original = forecast_table(a)
        a.loc[a.date == "2024-01-01", "AQI_eod"] = 490
        changed = forecast_table(a)
        before = original[original.feature_date == "2023-12-31"]
        after = changed[changed.feature_date == "2023-12-31"]
        np.testing.assert_allclose(before[FEATURES], after[FEATURES], equal_nan=True)
        self.assertNotEqual(before.y.iloc[0], after.y.iloc[0])

    def test_december_january_windows_do_not_wrap(self):
        calendar = pd.date_range("2023-12-27", "2024-01-03")
        weather = pd.DataFrame({c: np.arange(len(calendar), dtype=float) for c in seq_data.STEP_FEATURES}, index=calendar)
        rows = pd.DataFrame(dict(date=pd.to_datetime(["2023-12-31", "2024-01-01"]), city="Delhi", station_id="a", grid_lat=28., grid_lon=77.))
        raw = pd.DataFrame(dict(latitude=[28.], longitude=[77.]))
        with patch.object(seq_data, "YEAR_DAYS", calendar), patch.object(seq_data, "_weather_frame", return_value=weather), patch.object(seq_data.pd, "read_csv", return_value=raw):
            x = seq_data.build_windows(rows, max_window=3)
        np.testing.assert_equal(x[0, :, 0], [2, 3, 4])
        np.testing.assert_equal(x[1, :, 0], [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
