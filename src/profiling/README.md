# Profiling and execution checks

[`workloads/`](workloads/) contains the migrated BERT, GPT-2, ResNet-50,
ShuffleNet V2, VGG-16, and ViT scripts with their model dependencies and run
instructions. These actual inference workloads support NV/MPSE execution.

Archived measurements and profiles are published under
[`data/profiling/`](../../data/profiling/): all 2,852 CSV exports for the six
models, with original latency tables, raw NCU/NSYS exports, monitor records,
derived KS profiles, and checksums. See the dataset README for their formats.

`cinfo.c` and `cinfo.h` load kernel-profile CSVs into the runtime's lookup table.
`baseline.py` runs concurrent native CUDA (NV) or fixed-SM MPS (MPSE) contexts
using `vector_add.ptx`, verifies the GPU result, and records per-client latency.
This synthetic workload has no model weights or dataset dependencies.

From the repository root:

```bash
python3 src/profiling/baseline.py --config configs/NV.json
python3 src/profiling/baseline.py --config configs/MPSE.json
```

See [the run guide](../../docs/RUNNING.md) for prerequisites and measurement
scope. The included profiling archive is distinct from end-to-end experiment
results; the dataset README identifies its exact coverage.
