import sys
import unittest
from pathlib import Path
import json
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/dashboard'))
import app

class FollowthroughDashboardChecks(unittest.TestCase):
    def test_research_state_labels_pending_test(self):
        state=app.state()
        self.assertEqual(state['default_mode'],'next_day')
        r=state['research']
        self.assertEqual(r['status'],'complete')
        self.assertFalse(r['prospective_ensemble']['tested_on_fresh_data'])
        self.assertEqual(r['prospective_ensemble']['prospective_period'][0],'2026-10-13')
        self.assertEqual({a['city'] for a in r['attention_city']},{'Bengaluru','Chennai','Delhi','Hyderabad','Jaipur','Kolkata','Lucknow','Mumbai'})
        frame=state['modes']['next_day_2026']['rows']
        self.assertTrue(all(row['gru_pred'] is not None and row['pred'] is not None for row in frame))

    def test_downloads_match_saved_files_and_reject_unknown_names(self):
        client=app.app.test_client()
        files={'report':'Results_Report.txt','errors':'error_groups.csv','misses':'largest_errors.csv','attention':'attention_city.csv','deletion':'attention_deletion_city.csv','summary':'summary.json'}
        for key,name in files.items():
            response=client.get('/api/download/research/'+key)
            self.assertEqual(response.status_code,200)
            self.assertEqual(sha256(response.data).hexdigest(),sha256((ROOT/'results/followthrough/oct12-v1'/name).read_bytes()).hexdigest())
            response.close()
        self.assertEqual(client.get('/api/download/research/unknown').status_code,404)

if __name__=='__main__':unittest.main()
