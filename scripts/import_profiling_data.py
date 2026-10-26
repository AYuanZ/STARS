#!/usr/bin/env python3
"""Copy archived profiling CSVs with lossless compression and provenance.

No measurements are regenerated. Directly consumed latency/KS tables stay
uncompressed; other CSV exports are stored with deterministic gzip headers.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

MODELS = ('bert', 'gpt', 'resnet', 'shufflenet', 'vgg', 'vit')


def category(path):
    if path.suffix != '.csv':
        return {'.ncu-rep': 'nsight_compute_report', '.nsys-rep': 'nsight_systems_report',
                '.sqlite': 'nsight_systems_database', '.png': 'generated_plot'}.get(path.suffix, 'other')
    if 'all_result' in path.parts:
        return 'resource_monitor' if path.name == 'monitor.csv' else 'standalone_latency'
    if path.name == 'output_all_ncu.csv':
        return 'ncu_console_export'
    if path.name == 'raw_ncu.csv':
        return 'ncu_raw_metrics'
    if 'ks_nsys' in path.parts or path.name == 'output_nsys_gputrace.csv_gputrace.csv':
        return 'nsys_kernel_trace'
    if 'ks_nsys_ncu' in path.parts or path.name == 'output_ncu_nsys.csv':
        return 'joined_kernel_profile'
    if path.name.startswith('ks_summary_sm_'):
        return 'runtime_ks_summary'
    if path.name.startswith('partitioned_kernels_sm_'):
        return 'partitioned_kernel_profile'
    if path.name == 'kernel_segments_summary.csv':
        return 'kernel_segment_index'
    return 'processed_ncu_metrics'


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Original Exp/Profiling directory')
    parser.add_argument('--destination', type=Path, default=Path('data/profiling'))
    args = parser.parse_args()
    missing = [name for name in MODELS if not (args.source / name).is_dir()]
    if missing:
        parser.error(f'Missing model directories: {missing}')
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest, inventory, coverage = [], [], defaultdict(Counter)
    original_bytes = stored_bytes = 0
    for path in sorted(args.source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(args.source)
        model = relative.parts[0]
        kind = category(relative)
        size = path.stat().st_size
        entry = dict(source_relative_path=relative.as_posix(), model=model, category=kind,
                     source_bytes=size, status='not_published', stored_path='')
        if model not in MODELS:
            entry['status'] = 'outside_six_model_scope'
        elif path.suffix != '.csv':
            entry['status'] = 'binary_report_database_or_plot_not_published'
        else:
            plain = 'all_result' in relative.parts or kind in ('runtime_ks_summary', 'kernel_segment_index')
            target_relative = relative if plain else Path(str(relative) + '.gz')
            target = args.destination / target_relative
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite an existing data file: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open('rb') as source, target.open('wb') as output:
                destination = output if plain else gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0)
                try:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                        destination.write(block)
                finally:
                    if not plain:
                        destination.close()
            compressed_size = target.stat().st_size
            manifest.append(dict(stored_path=target_relative.as_posix(), source_relative_path=relative.as_posix(),
                                 model=model, category=kind, encoding='identity' if plain else 'gzip',
                                 source_bytes=size, stored_bytes=compressed_size,
                                 source_sha256=digest.hexdigest(), stored_sha256=hash_file(target)))
            entry.update(status='published', stored_path=target_relative.as_posix())
            coverage[model][kind] += 1
            original_bytes += size
            stored_bytes += compressed_size
            if len(manifest) % 250 == 0:
                print(f'Copied {len(manifest)} CSVs; stored {stored_bytes / 2**20:.1f} MiB', flush=True)
        inventory.append(entry)
    write_csv(args.destination / 'manifest.csv', list(manifest[0]), manifest)
    write_csv(args.destination / 'source_inventory.csv', list(inventory[0]), inventory)
    summary = dict(source_tree='Exp/Profiling', models=list(MODELS), csv_files=len(manifest),
                   source_bytes=original_bytes, stored_bytes=stored_bytes,
                   files_by_model_and_category={name: dict(sorted(counts.items())) for name, counts in sorted(coverage.items())},
                   inventory_status_counts=dict(Counter(row['status'] for row in inventory)))
    (args.destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
