from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/models'))
from forecast_development import development_splits,feature_table,WEATHER_RAW

class ForecastDevelopmentTests(unittest.TestCase):
    def frame(self):
        dates=pd.date_range('2024-08-01',periods=90)
        a=pd.DataFrame(dict(date=dates,station_id='s',city='city',station_name='station',station_lat=28.,station_lon=77.,AQI_eod=np.arange(90,dtype=float)+10,aqi_flat_day=False))
        for c in WEATHER_RAW: a[c]=np.arange(90,dtype=float)
        return a

    def test_2025_is_exclusively_test(self):
        dates=pd.date_range('2023-01-01','2025-12-31'); splits=development_splits(dates)
        self.assertEqual(len(splits['test']),365)
        self.assertEqual(sum(map(len,splits.values())),len(dates))
        for name in ('train','selection','calibration'):
            self.assertTrue((dates[splits[name]].year<2025).all())

    def test_future_targets_and_recent_weather_cannot_change_inputs(self):
        a=self.frame(); before,features=feature_table(a); date=pd.Timestamp('2024-10-01')
        a.loc[a.date>=date,'AQI_eod']=499
        a.loc[a.date>date-pd.Timedelta(days=7),WEATHER_RAW]=99999
        after,_=feature_table(a)
        cols=features['delayed_weather']
        np.testing.assert_equal(before.loc[before.date==date,cols].to_numpy(),after.loc[after.date==date,cols].to_numpy())
        self.assertNotEqual(before.loc[before.date==date,'y'].iloc[0],after.loc[after.date==date,'y'].iloc[0])

    def test_missing_calendar_day_never_becomes_previous_row(self):
        a=self.frame(); a=a[a.date!=pd.Timestamp('2024-09-30')]
        out,_=feature_table(a)
        self.assertFalse((out.date==pd.Timestamp('2024-10-01')).any())
        row=out[out.date==pd.Timestamp('2024-10-02')].iloc[0]
        self.assertTrue(np.isnan(row.aqi_l2))
        self.assertEqual(row.aqi_count3,2)

if __name__=='__main__': unittest.main()
