"""Host checks for workload portability; GPU inference is tested separately."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / 'src/profiling/workloads'
sys.path.insert(0, str(WORKLOADS))
import workload_support as support
import utils
import savelat


class WorkloadChecks(unittest.TestCase):
    def test_help_without_cuda_or_model_imports(self):
        for name in ('bert', 'gpt', 'resnet', 'shufflenet', 'vgg', 'vit'):
            result = subprocess.run([sys.executable, '-S', str(WORKLOADS / f'myt{name}.py'), '--help'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--model-path', result.stdout)

    def test_fixed_batch_and_slo(self):
        with patch.dict(os.environ, {'STARS_BATCH_MODE': 'fixed', 'STARS_FIXED_BATCH': '4',
                                     'STARS_FIXED_SLO_MS': '35'}):
            self.assertEqual(utils.gen_batch(), 4)
            self.assertEqual(utils.gen_rd_slo(6, 4, ''), 35)

    def test_profile_lookup_and_csv_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / 'profile.csv'
            profile.write_text('batch,SM(%),latency(ms)\n1,6,2.5\n4,6,8.5\n')
            self.assertEqual(utils.find_lat(6, 4, str(profile)), 8.5)
            with self.assertRaises(ValueError):
                utils.find_lat(8, 4, str(profile))
            output = Path(temporary) / 'out.csv'
            with patch.dict(os.environ, {'STARS_LATENCY_CSV': str(profile)}):
                savelat.save_sm_lat_tamp_csv(output, 'bert', 4, 6, 10, 20, 0, 2.5)
                savelat.save_sm_lat_tamp_csv(output, 'bert', 1, 6, 3, 20, 1, 2.5)
            with output.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]['ISO_Lat(ms)']), 8.5)
            self.assertEqual(float(rows[1]['ISO_Lat(ms)']), 2.5)

    def test_optional_barrier_and_timeout(self):
        support.wait_for_start('')
        with tempfile.TemporaryDirectory() as temporary:
            signal = Path(temporary) / 'start.json'
            signal.write_text(json.dumps({'message': 1}))
            support.wait_for_start(signal)
            signal.write_text('{')
            with patch.dict(os.environ, {'STARS_START_TIMEOUT': '0.01'}):
                with self.assertRaises(TimeoutError):
                    support.wait_for_start(signal)


if __name__ == '__main__':
    unittest.main()
