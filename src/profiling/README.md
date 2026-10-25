# Profiling and execution checks

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
scope. Original model profiling pipelines will be provided separately.
