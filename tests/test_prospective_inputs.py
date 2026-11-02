import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_prospective_inputs import aggregate_archive,VARIABLES

class ProspectiveInputChecks(unittest.TestCase):
    def hourly(self):
        return dict(hourly=dict(time=pd.date_range('2026-10-13',periods=24,freq='h').strftime('%Y-%m-%dT%H:%M').tolist(),
            **{v+'_previous_day1':[2.]*24 for v in VARIABLES}))

    def test_complete_archive_uses_actual_prior_issue_cutoff(self):
        r=aggregate_archive(self.hourly(),'Delhi').iloc[0]
        self.assertEqual(pd.Timestamp(r.forecast_available_before_utc),pd.Timestamp('2026-10-12T23:00:00Z'))
        self.assertEqual(r.fc_precipitation,48)
        self.assertEqual(r.fc_temperature_2m,2)

    def test_missing_hour_or_variable_is_not_filled(self):
        d=self.hourly();d['hourly']['temperature_2m_previous_day1'][3]=None
        self.assertTrue(np.isnan(aggregate_archive(d,'Delhi').iloc[0].fc_temperature_2m))
        d=self.hourly();d['hourly']={k:v[:-1] for k,v in d['hourly'].items()}
        row=aggregate_archive(d,'Delhi').iloc[0]
        self.assertTrue(all(np.isnan(row['fc_'+v]) for v in VARIABLES))

    def test_duplicate_archive_hour_is_rejected(self):
        d=self.hourly();d['hourly']['time'][-1]=d['hourly']['time'][0]
        with self.assertRaises(ValueError):aggregate_archive(d,'Delhi')

if __name__=='__main__':unittest.main()
