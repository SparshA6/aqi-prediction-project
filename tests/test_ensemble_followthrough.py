import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/models'))
from ensemble_followthrough import blend, paired_development, select_weight, prospective_ready, load_frozen
from build_followthrough_report import diagnostics

class EnsembleChecks(unittest.TestCase):
    def test_blend_endpoints_and_invalid_values(self):
        np.testing.assert_array_equal(blend([10,100],[90,200],0),[10,100])
        np.testing.assert_array_equal(blend([10,100],[90,200],1),[90,200])
        np.testing.assert_allclose(blend([10,100],[90,200],.5),[50,150])
        for a,b,w in [([1],[2],-1),([1],[2],2),([np.nan],[2],.5),([1,2],[3],.5)]:
            with self.assertRaises(ValueError):blend(a,b,w)

    def test_pairing_is_keyed_and_checks_cohorts(self):
        base=pd.DataFrame(dict(date=['2025-10-01','2025-10-02'],station_id=['a','a'],city=['Delhi','Delhi'],split=['selection']*2,y=[80,120],aqi_l1=[70,80]))
        a=base.assign(pred=[90,110],model='Frozen GRU-28days')
        b=base.assign(pred=[80,120],model='XGBoost-change-forecast-weather-d5').iloc[::-1]
        paired=paired_development(pd.concat([a,b]))
        np.testing.assert_array_equal(paired.pred_weather,paired.y)
        for bad in [pd.concat([a,b.iloc[:1]]),pd.concat([a,a.iloc[:1],b]),pd.concat([a,b.assign(y=0)])]:
            with self.assertRaises(ValueError):paired_development(bad)

    def test_selection_constraint_blocks_less_precise_weather(self):
        p=pd.DataFrame(dict(date=['2025-10-01']*3,split=['selection']*3,y=[170,110,110],reference=[100]*3,pred_gru=[160,110,110],pred_weather=[170,160,160]))
        weight,grid=select_weight(p)
        self.assertLess(weight,.8)
        self.assertFalse(bool(grid.iloc[-1].eligible))
        invalid=p.assign(date='2026-01-01')
        with self.assertRaises(ValueError):select_weight(invalid)

    def test_prospective_period_cannot_be_evaluated_early(self):
        m={'earliest_final_evaluation_utc':'2027-01-01'}
        self.assertFalse(prospective_ready(m,'2026-12-31T23:59:59Z'))
        self.assertTrue(prospective_ready(m,'2027-01-01T00:00:00Z'))

    def test_saved_freeze_and_selection_are_development_only(self):
        m=load_frozen()
        self.assertFalse(m['tested_on_fresh_data'])
        self.assertEqual(m['prospective_period'][0],'2026-10-13')
        self.assertLessEqual(m['selection_period'][1],'2025-10-31')

    def test_diagnostics_all_comparators_share_rows(self):
        r=pd.DataFrame(dict(date=['2026-01-01','2026-07-01'],station_id=['a','b'],city=['Delhi','Mumbai'],y=[150,70],reference=[100,80],pred=[130,75],gru_pred=[140,80]))
        metrics,misses=diagnostics(r)
        overall=metrics[metrics.group_type=='overall']
        self.assertEqual(set(overall.n),{2})
        self.assertEqual(len(overall),3)
        self.assertEqual(set(misses.model),{'Weather XGBoost','Frozen GRU','Persistence'})

if __name__=='__main__':unittest.main()
