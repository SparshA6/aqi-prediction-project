"""Tests for the prospective-pipeline guards in src/prospective_tracker.py.

These are the pure, no-network, no-filesystem checks that decide whether a prospective
forecast is allowed to be recorded: the reserved-window check (target_day) and the
deadline/consistency check (validate_capture). record() itself needs live CPCB/weather
snapshots and frozen model artifacts, so it is exercised end-to-end by running the
pipeline, not here.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prospective_tracker as tracker
from prospective_tracker import ground_summary, target_day, validate_capture


def test_target_day_accepts_dates_in_the_reserved_window():
    assert target_day("2026-10-13") == pd.Timestamp("2026-10-13")
    assert target_day("2026-12-31") == pd.Timestamp("2026-12-31")
    assert target_day("2026-11-15") == pd.Timestamp("2026-11-15")


@pytest.mark.parametrize("value", ["2026-10-12", "2027-01-01", "2026-01-01"])
def test_target_day_rejects_dates_outside_the_reserved_window(value):
    with pytest.raises(ValueError, match="reserved period"):
        target_day(value)


def test_target_day_rejects_timezone_aware_input():
    with pytest.raises(ValueError, match="unzoned"):
        target_day(pd.Timestamp("2026-11-01", tz="UTC"))


def test_target_day_rejects_non_midnight_input():
    with pytest.raises(ValueError, match="unzoned"):
        target_day("2026-11-01 06:00:00")


def test_ground_summary_separates_test_window_and_checks_prior_day(tmp_path):
    pd.DataFrame({"date": ["2026-10-12", "2026-10-13", "2026-10-30", "2026-10-31"],
                  "AQI_eod": [100.0, 110.0, 120.0, None], "aqi_flat_day": [False, False, False, False]}).to_csv(tmp_path / "aligned_dataset.csv", index=False)
    period = ["2026-10-13", "2026-12-31"]
    ready = ground_summary(tmp_path, period, "2026-10-31")
    assert ready == dict(usable_station_days_in_period=2, latest_observed_aqi="2026-10-30", target_has_prior_day_aqi=True)
    assert ground_summary(tmp_path, period, "2026-11-03")["target_has_prior_day_aqi"] is False


def test_weather_hours_are_incomplete_until_the_last_hour_before_target():
    time = pd.date_range("2026-11-03", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    data = {"hourly": {"time": time, **{v + "_previous_day1": [1.0] * 24 for v in tracker.VARIABLES}}}
    response = Mock(headers={}, url="u")
    response.json.return_value = data
    stations = pd.DataFrame([{"city": "Delhi", "lat": 28.6, "lon": 77.2}])
    for now, status in [("2026-11-02T22:59:00Z", "waiting_for_complete_earlier_runs"), ("2026-11-02T23:00:00Z", "ready")]:
        with tempfile.TemporaryDirectory() as folder, patch.object(tracker, "OUT", Path(folder)), \
                patch.object(tracker, "now_utc", return_value=pd.Timestamp(now)), \
                patch.object(tracker.pd, "read_csv", return_value=stations), patch.object(tracker.requests, "get", return_value=response):
            directory = tracker.collect_weather("2026-11-03")
            assert json.loads((directory / "manifest.json").read_text())["complete_cities"] == (status == "ready")


def _manifest(target, retrieved_utc):
    return {"target_date": str(target.date()), "retrieved_utc": retrieved_utc}


def _weather(issue_isos):
    return pd.DataFrame({"forecast_available_before_utc": issue_isos})


def test_validate_capture_passes_when_everything_is_before_the_deadline():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-04T10:00:00Z")
    manifest = _manifest(target, "2026-11-04T09:00:00+00:00")
    weather = _weather(["2026-11-04T08:00:00+00:00"])
    validate_capture(target, now, weather, manifest)  # must not raise


def test_validate_capture_rejects_after_the_target_has_begun():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-05T00:00:01Z")  # past UTC midnight of the target day
    manifest = _manifest(target, "2026-11-04T09:00:00+00:00")
    weather = _weather(["2026-11-04T08:00:00+00:00"])
    with pytest.raises(ValueError, match="Deadline passed"):
        validate_capture(target, now, weather, manifest)


def test_validate_capture_rejects_weather_captured_after_target_began():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-04T10:00:00Z")
    manifest = _manifest(target, "2026-11-05T00:30:00+00:00")  # captured after midnight
    weather = _weather(["2026-11-04T08:00:00+00:00"])
    with pytest.raises(ValueError, match="captured after target began"):
        validate_capture(target, now, weather, manifest)


def test_validate_capture_rejects_a_weather_snapshot_for_the_wrong_day():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-04T10:00:00Z")
    manifest = {"target_date": "2026-11-06", "retrieved_utc": "2026-11-04T09:00:00+00:00"}
    weather = _weather(["2026-11-04T08:00:00+00:00"])
    with pytest.raises(ValueError, match="another day"):
        validate_capture(target, now, weather, manifest)


def test_validate_capture_rejects_an_issue_time_at_or_after_the_target_day():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-04T10:00:00Z")
    manifest = _manifest(target, "2026-11-04T09:00:00+00:00")
    weather = _weather(["2026-11-05T00:00:00+00:00"])  # issue time not before the target
    with pytest.raises(ValueError, match="not available before target"):
        validate_capture(target, now, weather, manifest)


def test_validate_capture_rejects_an_issue_time_in_the_future_relative_to_now():
    target = pd.Timestamp("2026-11-05")
    now = pd.Timestamp("2026-11-04T10:00:00Z")
    manifest = _manifest(target, "2026-11-04T09:00:00+00:00")
    weather = _weather(["2026-11-04T11:00:00+00:00"])  # after `now`, i.e. not actually known yet
    with pytest.raises(ValueError, match="not available before target"):
        validate_capture(target, now, weather, manifest)
