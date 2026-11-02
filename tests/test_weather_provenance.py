"""The weather snapshot must retain auditable retrieval evidence."""
from pathlib import Path
from unittest.mock import Mock, patch
import json
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import prospective_tracker as tracker


class WeatherProvenanceTests(unittest.TestCase):
    def test_snapshot_hashes_response_and_retrieval_audit(self):
        time = pd.date_range('2026-10-13', periods=24, freq='h').strftime('%Y-%m-%dT%H:%M').tolist()
        data = {'hourly': {'time': time, **{v+'_previous_day1': [1.0]*24 for v in tracker.VARIABLES}}}
        response = Mock()
        response.json.return_value = data
        response.headers = {'Date': 'Mon, 12 Oct 2026 23:30:00 GMT'}
        response.url = 'https://previous-runs-api.open-meteo.com/v1/forecast?test=1'
        now = pd.Timestamp('2026-10-12T23:30:00Z')
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(tracker, 'OUT', Path(folder)), \
             patch.object(tracker, 'now_utc', return_value=now), \
             patch.object(tracker.pd, 'read_csv', return_value=pd.DataFrame([{'city': 'Delhi', 'lat': 28.6, 'lon': 77.2}])), \
             patch.object(tracker.requests, 'get', return_value=response) as get:
            directory = tracker.collect_weather('2026-10-13')
            manifest = json.loads((directory/'manifest.json').read_text())
            audit = json.loads((directory/'retrieval_audit.json').read_text())
            exact = json.loads((directory/'single_run_audit.json').read_text())
            self.assertEqual(manifest['complete_cities'], 1)
            self.assertEqual(manifest['retrieval_audit_sha256'], tracker.digest(directory/'retrieval_audit.json'))
            self.assertEqual(manifest['exact_run_audit_sha256'], tracker.digest(directory/'single_run_audit.json'))
            self.assertEqual(audit[0]['provider_http_date'], response.headers['Date'])
            self.assertEqual(audit[0]['raw_sha256'], manifest['source_sha256']['Delhi.json'])
            self.assertIsNone(audit[0]['provider_run_id'])
            self.assertEqual(exact[0]['run_initialized_utc'], '2026-10-12T12:00:00+00:00')
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args.kwargs['params']['run'], '2026-10-12T12:00')


if __name__ == '__main__':
    unittest.main()
