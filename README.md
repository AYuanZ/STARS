# STARS

STARS is a research project for efficient spatio-temporal GPU resource
sharing in concurrent edge AI inference workloads.

The project explores how profiling, interference modeling, kernel-level
execution control, and runtime scheduling can be combined to improve GPU
utilization while satisfying the latency requirements of heterogeneous
inference tasks.

> **Release status:** This repository is currently under active preparation.
> We have updated the publicly available codebase to include the STARS
> implementation, CUDA interception components, interference predictor,
> scheduling algorithms, and baseline configurations. We will also provide
> detailed documentation of compiler versions, software dependencies, CUDA
> environment, model configurations and weight sources, along with profiling
> scripts, experimental configurations, raw data, and analysis scripts.
>
> This snapshot includes build instructions, a verified environment, and
> runnable NV/MPSE configurations. The included GPU examples are synthetic
> execution checks, not a reproduction of the inference results. The remaining
> experimental artifacts will be released incrementally.

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

Run the native CUDA and fixed-SM MPS baselines on a machine with a working
NVIDIA driver; MPSE also needs `nvidia-cuda-mps-control` on `PATH`:

```bash
python3 src/profiling/baseline.py --config configs/NV.json
python3 src/profiling/baseline.py --config configs/MPSE.json
```

Each command launches two clients, checks GPU results, and writes per-client
latency CSVs under a new `results/` directory. MPSE creates and shuts down its
own isolated MPS controller. Change `sm_count` in `configs/MPSE.json` to adjust
the requested SM limit; the CSV records the effective limit reported by CUDA.
Neither baseline needs model weights, PyTorch, the STARS daemon, or `LD_PRELOAD`.

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
├── tests/            # Host integration checks
└── src/              # STARS implementation
    ├── profiling/    # Workload and GPU resource profiling
    ├── modeling/     # Performance and interference modeling
    ├── scheduler/    # SLO-aware scheduling and resource allocation
    └── runtime/      # GPU sharing runtime and CUDA interception
```

Runtime sources are available in [`src/runtime/core/`](src/runtime/core/).
Original third-party copyright notices are retained; see [LICENSE](LICENSE).
