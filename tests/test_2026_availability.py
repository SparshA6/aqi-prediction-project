from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import finish_forecast_2026 as worker

class AvailabilityTests(unittest.TestCase):
    def test_retry_must_complete_before_excluding(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'retry.log';p.write_text('Downloading January')
            self.assertIsNone(worker.completed_retry_exclusions(p))

    def test_exclusion_requires_audited_retry_and_zero_exports(self):
        metadata=pd.DataFrame([dict(station_id='s',station_name='Station',city='Delhi')])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'retry.log';p.write_text('                    COMPLETE\n[Delhi] Station: 0/9 months')
            with patch.object(worker.pd,'read_csv',return_value=metadata),patch.object(worker,'files_for',return_value=[]):
                self.assertEqual(worker.completed_retry_exclusions(p)[0]['station_id'],'s')
            with patch.object(worker.pd,'read_csv',return_value=metadata),patch.object(worker,'files_for',return_value=[(1,Path('a.xlsx'))]):
                with self.assertRaisesRegex(ValueError,'Partial or duplicate'):worker.completed_retry_exclusions(p)
            p.write_text('                    COMPLETE\n')
            with patch.object(worker.pd,'read_csv',return_value=metadata),patch.object(worker,'files_for',return_value=[]):
                with self.assertRaisesRegex(ValueError,'not retried'):worker.completed_retry_exclusions(p)

if __name__=='__main__':unittest.main()
