"""CSV writers adapted from Myshare/evaluation/savelat.py.

Keep the original field names for analysis compatibility. Lock the file so
concurrent model workers cannot race when creating the header or adding rows.
"""
import csv
import fcntl
import os
from pathlib import Path


def append_row(path, header, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+', newline='') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0, os.SEEK_END)
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(header)
        writer.writerow(row)
        file.flush()


def save_sm_lat_tamp_csv(csv_file, modelname, batchsize, sm, latency, slo, timestamp, isolat):
    profile = os.environ.get('STARS_LATENCY_CSV')
    if profile:
        from utils import find_lat
        isolat = find_lat(sm, batchsize, profile)
    append_row(csv_file,
               ['model', 'batch', 'SM(%)', 'latency(ms)', 'ISO_Lat(ms)', 'SLO(ms)', 'timetamp'],
               [modelname, batchsize, sm, latency, isolat, slo, timestamp])


def save_task_sm_lat_csv(csv_file, batchsize, sm, latency):
    append_row(csv_file, ['batch', 'SM(%)', 'latency(ms)'], [batchsize, sm, latency])
