"""Batch and SLO sampling adapted from Myshare/evaluation/utils.py."""
from functools import lru_cache
import csv
import math
import os
import random


def gen_batch(select=1):
    mode = os.environ.get('STARS_BATCH_MODE', 'fixed1')
    if mode == 'fixed':
        return int(os.environ['STARS_FIXED_BATCH'])
    if mode == 'paper':
        values, weights = ([1, 2, 4, 8], [0.2, 0.5, 0.2, 0.1]) if select == 2 else (
            [1, 2, 4, 8, 16], [0.2, 0.3, 0.4, 0.1, 0.05])
        return random.choices(values, weights=weights, k=1)[0]
    if mode != 'fixed1':
        raise ValueError(f'Unsupported STARS_BATCH_MODE={mode!r}')
    return 1


@lru_cache(maxsize=8)
def load_profile(path):
    required = {'batch', 'SM(%)', 'latency(ms)'}
    with open(path, newline='') as file:
        reader = csv.DictReader(file)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path} must contain columns {sorted(required)}')
        return [{key: float(row[key]) for key in required} for row in reader]


def find_lat(sms, batch_size, csv_file):
    table = load_profile(csv_file)
    rows = [row for row in table if row['batch'] == batch_size and row['SM(%)'] == sms]
    if not rows:
        raise ValueError(f'No latency entry for batch={batch_size}, SM count={sms} in {csv_file}')
    latency = rows[0]['latency(ms)']
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError('Profile latencies must be positive and finite')
    return latency


def gen_rd_slo(sms, bs, lat_csv_path):
    fixed = os.environ.get('STARS_FIXED_SLO_MS')
    if fixed is not None:
        return float(fixed)
    latency = find_lat(sms, bs, lat_csv_path)
    # Preserve the original request-loop SLO multiplier distribution.
    factor = random.choices([1.5, 2, 4, 10], weights=[0.25, 0.5, 0.2, 0.05], k=1)[0]
    return int(round(latency * factor))
