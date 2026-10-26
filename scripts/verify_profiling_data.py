#!/usr/bin/env python3
"""Verify stored and decompressed SHA-256 hashes and inspect profile coverage."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

MODELS = ('bert', 'gpt', 'resnet', 'shufflenet', 'vgg', 'vit')
KS_COLUMNS = ['ks_id', 'num_kernels', 'duration_ns', 'avg_sm_throughput',
              'avg_dram_throughput', 'avg_l2_throughput', 'now_duration_ns', 'solo_duration_ns']


def digest_stream(file):
    digest = hashlib.sha256()
    length = 0
    for block in iter(lambda: file.read(1024 * 1024), b''):
        length += len(block)
        digest.update(block)
    return length, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/profiling'))
    parser.add_argument('--write-report', type=Path, help='Optional path for a machine-readable coverage report')
    parser.add_argument('--hashes-only', action='store_true', help='Verify another data manifest without the six-model schema checks')
    args = parser.parse_args()
    with (args.root / 'manifest.csv').open(newline='') as file:
        manifest = list(csv.DictReader(file))
    assert len({row['stored_path'] for row in manifest}) == len(manifest), 'Duplicate manifest entries'
    directories = [p for p in args.root.iterdir() if p.is_dir()] if args.hashes_only else [args.root / model for model in MODELS]
    found = {p.relative_to(args.root).as_posix() for directory in directories
             for p in directory.rglob('*') if p.is_file()}
    assert found == {row['stored_path'] for row in manifest}, 'Data files and manifest disagree'
    source_bytes = 0
    for index, row in enumerate(manifest, 1):
        path = args.root / row['stored_path']
        with path.open('rb') as file:
            length, digest = digest_stream(file)
        assert length == int(row['stored_bytes']), path
        assert digest == row['stored_sha256'], path
        opener = gzip.open if row['encoding'] == 'gzip' else open
        with opener(path, 'rb') as file:
            length, digest = digest_stream(file)
        assert length == int(row['source_bytes']), path
        assert digest == row['source_sha256'], path
        source_bytes += length
        if index % 500 == 0:
            print(f'Verified {index}/{len(manifest)} files', flush=True)
    if args.hashes_only:
        print(f'Verified {len(manifest)} data files ({source_bytes} original bytes).', flush=True)
        return
    profiles = {}
    for model in MODELS:
        path = args.root / model / 'all_result' / f'{model}.csv'
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
        keys = Counter((int(row['batch']), int(float(row['SM(%)']))) for row in rows)
        assert all(math.isfinite(float(row['latency(ms)'])) and float(row['latency(ms)']) > 0 for row in rows), path
        profiles[model] = dict(rows=len(rows), unique_batch_sm_pairs=len(keys),
                               repeated_batch_sm_pairs=sum(n > 1 for n in keys.values()),
                               batches=sorted({key[0] for key in keys}),
                               sm_counts=sorted({key[1] for key in keys}))
    ks_files = 0
    for row in manifest:
        if row['category'] != 'runtime_ks_summary':
            continue
        path = args.root / row['stored_path']
        with path.open(newline='') as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == KS_COLUMNS, path
            rows = list(reader)
        assert rows, path
        total = sum(int(record['duration_ns']) for record in rows)
        cumulative = 0
        for record in rows:
            assert int(record['num_kernels']) > 0 and int(record['duration_ns']) > 0, path
            cumulative += int(record['duration_ns'])
            assert int(record['now_duration_ns']) == total, path
            assert int(record['solo_duration_ns']) == cumulative, path
        ks_files += 1
    report = dict(verified_csv_files=len(manifest), verified_source_bytes=source_bytes,
                  runtime_ks_summary_files=ks_files, standalone_latency_profiles=profiles,
                  ks_duration_contract=dict(now_duration_ns='total kernel-segment duration in this profile',
                                            solo_duration_ns='cumulative duration through the current segment'))
    if args.write_report:
        args.write_report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
