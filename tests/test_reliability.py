"""Regression checks for experiment preservation and temporal validation."""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

import numpy as np
import pandas as pd
import torch

from check_results import coverage_errors
from combine_with_aqi import aqi_band
from data import band
import runs
from seq_models import make_net
from validation import temporal_validation_split


class ReliabilityTests(unittest.TestCase):
    def test_windows_are_disjoint_even_with_missing_dates(self):
        dates = pd.date_range("2023-01-01", periods=90).delete([10, 50, 65])
        dates = dates.repeat(3)
        fit, val = temporal_validation_split(dates, 7)
        fit_days = {d - pd.Timedelta(days=k) for d in dates[fit] for k in range(7)}
        val_days = {d - pd.Timedelta(days=k) for d in dates[val] for k in range(7)}
        self.assertFalse(fit_days & val_days)
        self.assertTrue(fit.any() and val.any())
        self.assertFalse((fit & val).any())

    def test_small_training_period_fails_clearly(self):
        with self.assertRaises(ValueError):
            temporal_validation_split(pd.date_range("2023-01-01", periods=3), 7)

    def test_fractional_bands_have_no_gaps(self):
        for value, category in [(50.5, "Satisfactory"), (100.5, "Moderate"), (200.5, "Poor"), (300.5, "Very Poor"), (400.5, "Severe")]:
            self.assertEqual(aqi_band(value), category)
            self.assertIn(category, band(value))
        self.assertIsNone(aqi_band(-1))

    def test_run_cannot_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            dataset = root / "aligned.csv"
            dataset.write_text("unchanged data")
            gee = root / "gee"
            gee.mkdir()
            (gee / "Delhi.csv").write_text("weather")
            historical = root / "results" / "baselines"
            historical.mkdir(parents=True)
            sentinel = historical / "metrics_chrono.csv"
            sentinel.write_text("historic results")
            args = argparse.Namespace(protocol="chrono", quick=True, run_name="test", folds=None, models=["Ridge"], sets=["SW"])
            df = pd.DataFrame({"station_id": ["a"], "city": ["Delhi"]})
            with patch.object(runs, "ROOT", root), patch.object(runs, "ALIGNED_CSV", dataset), patch.object(runs, "GEE_DIR", gee):
                out = runs.start_run(args, "baselines", df, [("chrono", [], {})])
                self.assertEqual(out.parent.name, "smoke")
                with self.assertRaises(FileExistsError):
                    runs.start_run(args, "baselines", df, [("chrono", [], {})])
            self.assertEqual(sentinel.read_text(), "historic results")

    def test_incomplete_and_duplicate_folds_are_rejected(self):
        m = pd.DataFrame([dict(split="test", subset="has_lag1", level="overall", fold="loco:Delhi", feature_set="SW", model="TA", RMSE=10., MAE=8., n=100)])
        errors = coverage_errors(m, "loco", folds={"loco:Delhi", "loco:Mumbai"})
        self.assertTrue(any("Mumbai" in e for e in errors))
        self.assertFalse(coverage_errors(m, "loco", folds={"loco:Delhi"}))
        self.assertTrue(coverage_errors(pd.concat([m, m]), "loco", folds={"loco:Delhi"}))

    def test_uniform_attention_preserves_tensor_dtype(self):
        for name in ["Uniform", "FA"]:
            net = make_net(name, 13).double().eval()
            with torch.no_grad():
                y, alpha, _ = net(torch.zeros(2, 7, 13, dtype=torch.float64), return_attn=True)
            self.assertEqual(alpha.dtype, torch.float64)
            self.assertTrue(torch.isfinite(y).all())

    def test_day_replacement_respects_row_specific_order(self):
        from audit_attention import replace_days
        x = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(float)
        y = replace_days(x, np.array([[2, 1, 0], [0, 2, 1]]), 1, np.array([-1., -2.]))
        np.testing.assert_equal(y[0, 2], [-1., -2.])
        np.testing.assert_equal(y[1, 0], [-1., -2.])
        np.testing.assert_equal(y[0, :2], x[0, :2])


if __name__ == "__main__":
    unittest.main()
