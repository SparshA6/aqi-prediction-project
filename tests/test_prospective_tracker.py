"""Tests for the prospective-pipeline guards in src/prospective_tracker.py.

These are the pure, no-network, no-filesystem checks that decide whether a prospective
forecast is allowed to be recorded: the reserved-window check (target_day) and the
deadline/consistency check (validate_capture). record() itself needs live CPCB/weather
snapshots and frozen model artifacts, so it is exercised end-to-end by running the
pipeline, not here.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prospective_tracker import target_day, validate_capture


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
