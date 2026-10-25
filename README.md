# STARS

STARS is a research project for efficient spatio-temporal GPU resource
sharing in concurrent edge AI inference workloads.

The project explores how profiling, interference modeling, kernel-level
execution control, and runtime scheduling can be combined to improve GPU
utilization while satisfying the latency requirements of heterogeneous
inference tasks.

> **Release status:** The released code includes the STARS runtime, CUDA
> interception, interference prediction, scheduling algorithms, six inference
> workloads.

## Build and run

On Linux with GCC and GNU Make:

```bash
make -j4
make schedulers
make check
```

The artifacts are `build/libstars.so`, `build/stars-scheduler`, and
`build/starsctl`. The host build does not require a CUDA toolkit. `make check`
checks scheduler/CLI communication and configuration handling without a GPU.

The six model scripts are in [`src/profiling/workloads/`](src/profiling/workloads/):
[`mytbert.py`](src/profiling/workloads/mytbert.py),
[`mytgpt.py`](src/profiling/workloads/mytgpt.py),
[`mytresnet.py`](src/profiling/workloads/mytresnet.py),
[`mytshufflenet.py`](src/profiling/workloads/mytshufflenet.py),
[`mytvgg.py`](src/profiling/workloads/mytvgg.py), and
[`mytvit.py`](src/profiling/workloads/mytvit.py).
NV and MPSE select how these models execute; they are not model names.

With CUDA-enabled PyTorch and the [model dependencies](src/profiling/workloads/requirements.txt)
installed, run actual inference in native CUDA or fixed-SM MPS mode:

```bash
PYTHON=python bash scripts/run_model.sh resnet NV --slo-ms 100 -q 10
PYTHON=python bash scripts/run_model.sh resnet MPSE --slo-ms 100 -sm 6 -q 10
```

Replace `resnet` with `bert`, `gpt`, `shufflenet`, `vgg`, or `vit` to run the
other models. Default pretrained weights use framework caches and may download
missing files; pass `--model-path` to use a local checkpoint. The fixed 100 ms
SLO above is an example deadline. Use `--latency-csv` for profile-based SLOs.
MPSE also requires `nvidia-cuda-mps-control` on `PATH`; the launcher manages
an isolated controller. See the [model run guide](src/profiling/workloads/README.md)
for input shapes, weight selection, original research branches, and validation.

The separate `src/profiling/baseline.py` and `configs/NV.json` / `MPSE.json`
remain available as optional vector-add checks of the CUDA/MPS environment.
They do not run any of the six inference models.

See [the run guide](docs/RUNNING.md) for requirements, source mapping, runtime
integration, and the scope of the validation performed on this snapshot.
See [the code guide](docs/CODE_GUIDE.md) for terminology, the scheduling data
flow, profile formats, and compatibility notes.

## Overview

Running multiple AI inference workloads on the same GPU introduces several
challenges:

- Workloads have different latency objectives and resource demands.
- Concurrent kernels may interfere with each other unpredictably.
- Static resource allocation can cause GPU underutilization.
- Temporal scheduling alone may not provide sufficient isolation.
- Fine-grained GPU sharing requires both spatial and temporal coordination.

STARS addresses these challenges through the following workflow:

1. Profile the runtime and GPU resource characteristics of each workload.
2. Analyze kernel behavior and construct an interference model.
3. Estimate the performance of candidate resource-sharing decisions.
4. Coordinate spatial resource allocation and temporal execution.
5. Dynamically schedule concurrent inference workloads.
6. Evaluate latency, SLO attainment, throughput, and GPU utilization.

## System Architecture

```text
AI Inference Workloads
          |
          v
  Offline Profiling
          |
          v
Kernel and Resource Characterization
          |
          v
  Interference Modeling
          |
          v
Scheduling and Resource Allocation
          |
          v
    GPU Sharing Runtime
          |
          v
 Evaluation and Analysis
```

## Repository Structure

```text
STARS/
├── Makefile          # Complete host build and checks
├── configs/          # Example model, hardware, and scenario configurations
├── docs/             # Run guide and release provenance
├── scripts/          # Actual-model launchers
├── tests/            # Host integration checks
└── src/              # STARS implementation
    ├── profiling/    # Workload and GPU resource profiling
    ├── modeling/     # Performance and interference modeling
    ├── scheduler/    # SLO-aware scheduling and resource allocation
    └── runtime/      # GPU sharing runtime and CUDA interception
```

Runtime sources are available in [`src/runtime/core/`](src/runtime/core/).
Original third-party copyright notices are retained; see [LICENSE](LICENSE).
