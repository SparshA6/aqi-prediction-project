import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/dashboard'))
sys.path.insert(0,str(ROOT/'src/models'))
import app
from forecast_diagnostics import summarize

class DashboardExtensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=app.app.test_client()

    def test_bad_city_inputs(self):
        for payload in [[],{'city':'Unknown','observed_date':'2025-10-31'},
                        {'city':'Delhi','observed_date':'2027-01-01'},
                        {'city':'Delhi','observed_date':'2025-10-31T00:00:00+05:30'}]:
            self.assertEqual(self.client.post('/api/city-forecast',json=payload).status_code,400)

    def test_city_forecast_mean_and_future_exclusion(self):
        body={'city':'Bengaluru','observed_date':'2025-10-31'}
        response=self.client.post('/api/city-forecast',json=body)
        self.assertEqual(response.status_code,200)
        a=response.get_json()
        self.assertEqual(a['target_date'],'2025-11-01')
        self.assertEqual(a['available_stations'],len(a['stations']))
        self.assertLessEqual(a['available_stations'],a['expected_stations'])
        self.assertAlmostEqual(a['prediction'],np.mean([r['prediction'] for r in a['stations']]),places=7)
        original=app.read
        def altered(path):
            frame=original(path)
            if Path(path).name=='aligned_dataset.csv':
                frame.loc[pd.to_datetime(frame.date)>'2025-10-31','AQI_eod']=500
            return frame
        with patch.object(app,'read',side_effect=altered):
            b=self.client.post('/api/city-forecast',json=body).get_json()
        self.assertEqual(a,b)

    def test_environment_selection_not_test_selection(self):
        s=app.state();m=pd.read_csv(ROOT/'results/year_holdout/runs/2023-to-2024-v1/metrics.csv')
        v=m[(m['mode']=='same_day')&(m.split=='selection')&m.model.str.endswith('-env')]
        self.assertEqual(s['modes']['same_day_env']['model'],v.sort_values(['RMSE','model']).iloc[0].model)
        self.assertTrue(all('reference' in r for r in s['modes']['next_day']['rows']))

    def test_packed_state_keeps_every_row(self):
        full=app.state();packed=app.packed_state()
        stations=packed['stations']
        for name,mode in full['modes'].items():
            p=packed['modes'][name];c=p['columns']
            self.assertNotIn('rows',p)
            self.assertEqual(len(c['day']),len(mode['rows']))
            for i in [0,len(mode['rows'])//2,len(mode['rows'])-1]:
                row=mode['rows'][i]
                self.assertEqual(p['days'][c['day'][i]],row['date'][:10])
                self.assertEqual(stations[c['station'][i]]['station_id'],row['station_id'])
                self.assertEqual(stations[c['station'][i]]['city'],row['city'])
                for key in ['y','pred','reference','gru_pred','lower','upper']:
                    if key in row: self.assertAlmostEqual(c[key][i],row[key],places=3)
            self.assertAlmostEqual(float(np.sqrt(np.mean((np.array(c['pred'])-np.array(c['y']))**2))),mode['rmse'],places=3)
        self.assertLess(len(json.dumps(packed)),len(json.dumps(full))/3)

    def test_page_data_used_by_new_sections(self):
        s=app.state()
        self.assertEqual(len(s['attention_stability']),36)
        self.assertTrue(all(np.isfinite([r['lat'],r['lon']]).all() for r in s['stations']))
        page=(ROOT/'src/dashboard/index.html').read_text(encoding='utf-8')
        self.assertEqual(page.count('/* SNAPSHOT */'),1)
        self.assertIn("/api/dashboard",page)

    def test_export_links_public_files_only_when_asked(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(app,'packed_state',return_value={}):
            path=Path(folder)/'index.html'
            with patch.object(sys,'argv',['app.py','--export',str(path)]): app.main()
            self.assertNotIn('window.publicFiles=true;',path.read_text(encoding='utf-8'))
            with patch.object(sys,'argv',['app.py','--export',str(path),'--public-files']): app.main()
            page=path.read_text(encoding='utf-8')
            self.assertIn('window.snapshot={};window.publicFiles=true;',page)
            for name in ['AQI_Local_Demo.zip','AQI_Methodology_Submission.pdf','AQI_Results_and_Discussion_Submission.pdf']:
                self.assertIn(f'href="{name}"',page)

    def test_forecast_values_give_readable_errors(self):
        body={'station_id':'alipur_delhi','observed_date':'2025-10-31','latest_aqi':100}
        for values in [[None]*27,['x']*27]:
            response=self.client.post('/api/forecast',json={**body,'previous_aqi':values})
            self.assertEqual(response.status_code,400)
            self.assertEqual(response.get_json()['error'],'Each previous AQI must be a number')

    def test_spike_direction_and_empty_denominators(self):
        df=pd.DataFrame(dict(y=[170,30,110],reference=[100,100,100],pred=[160,160,160]))
        result=summarize(df,50)
        self.assertEqual(result['sharp_changes'],2)
        self.assertEqual(result['direction_correct_detections'],1)
        self.assertEqual(result['recall'],.5)
        self.assertAlmostEqual(result['precision'],1/3)
        quiet=summarize(pd.DataFrame(dict(y=[100],reference=[100],pred=[100])))
        self.assertIsNone(quiet['recall']);self.assertIsNone(quiet['precision'])

if __name__=='__main__': unittest.main()
