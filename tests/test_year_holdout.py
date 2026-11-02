"""Guard the untouched-year evaluation and cross-year forecast history."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/models'))
from year_holdout import split_years, forecast_windows, normalize_windows, paired_week_bootstrap

class YearHoldoutTests(unittest.TestCase):
    def test_year_2024_is_exclusively_test(self):
        dates=pd.date_range('2023-01-01','2024-12-31')
        splits=split_years(dates)
        self.assertEqual(len(splits['test']),366)
        for name in ['train','selection','calibration']:
            self.assertTrue((dates[splits[name]].year==2023).all())
            self.assertFalse(set(splits[name]) & set(splits['test']))
        self.assertEqual(sum(map(len,splits.values())),len(dates))

    def test_future_target_change_does_not_enter_window(self):
        dates=pd.date_range('2023-12-25','2024-01-03')
        a=pd.DataFrame(dict(date=dates,station_id='a',AQI_eod=np.arange(len(dates),dtype=float),aqi_flat_day=False))
        rows=pd.DataFrame(dict(date=pd.to_datetime(['2024-01-01','2024-01-02']),station_id='a',lat=28.,lon=77.))
        before=forecast_windows(a,rows)
        a.loc[a.date>='2024-01-01','AQI_eod']=499
        after=forecast_windows(a,rows)
        np.testing.assert_equal(before[0],after[0])
        self.assertNotEqual(before[1,-1,0],after[1,-1,0])
        np.testing.assert_equal(before[0,:,0],np.arange(7))

    def test_preprocessing_uses_only_supplied_fitting_rows(self):
        fit=np.array([[[1.,0.],[3.,0.]]],dtype=np.float32)
        data=np.concatenate([fit,np.array([[[100.,1.],[np.nan,1.]]],dtype=np.float32)])
        transformed,mu,sd=normalize_windows(fit,data)
        np.testing.assert_equal(mu,[2.,0.])
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed[1,1,0],0)

    def test_paired_reference_identity_has_zero_gain(self):
        rows=pd.DataFrame(dict(date=pd.date_range('2024-01-01',periods=20),y=np.arange(20),pred=np.arange(20)+2,reference=np.arange(20)+2))
        result=paired_week_bootstrap(rows,100)
        self.assertEqual([result[k] for k in ['gain','lower','upper']],[0.,0.,0.])

if __name__=='__main__': unittest.main()
