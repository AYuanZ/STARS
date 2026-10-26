# Code guide

## Execution paths

The repository contains three execution paths:

- `src/profiling/workloads/myt*.py` contains the six migrated inference model
  workloads: BERT, GPT-2, ResNet-50, ShuffleNet V2, VGG-16, and ViT. Use
  `scripts/run_model.sh` for their NV/MPSE execution modes. These scripts load
  pretrained models and generate input tensors as in the original source.
  Their [model guide](../src/profiling/workloads/README.md) documents checkpoint
  selection, parameters, dependencies, and GPU validation.
- `src/profiling/baseline.py` runs the independent NV/MPSE GPU checks. NV uses
  ordinary CUDA contexts; MPSE uses MPS contexts with an explicit SM-count
  affinity. The runner requires neither the STARS daemon nor its preload
  library. Its vector-add workload checks execution and reports latency; it
  does not load an inference model.
- `src/runtime/core/` and `src/scheduler/` implement the STARS research runtime.
  The interception library exchanges workload metadata with the scheduler and
  uses kernel profiles when choosing execution windows and SM allocations.
  The migrated `Our` branches retain application-side metadata transfers;
  full STARS execution still requires compatible profiles and MPS. See
  [RUNNING.md](RUNNING.md) for setup details.

NV means native CUDA execution in the supplied configurations. MPSE means the
fixed-SM MPS baseline. Neither name selects one of the temporal scheduler
executables built by `make schedulers`.

## Terminology and data flow

| Name | Meaning in this snapshot |
| --- | --- |
| SM | GPU streaming multiprocessor; `smcount` is an allocation count |
| KS | Kernel segment, the unit represented by a kernel-profile record |
| `ks_id` / KSID | Kernel-segment identifier, with allocation and batch information encoded for profile lookup |
| `KS_L` / `ks_len` | Selected execution window measured in kernel segments |
| `bs` | Inference batch size |
| SLO | Per-request latency objective |
| Colocated tasks | Tasks sharing the GPU at the same time |
| TQ | Legacy scheduler time quantum |

The profile loader in `src/profiling/cinfo.c` encodes its lookup key as
`ks_id * 10000 + smcount * 100 + bs`. The SM and batch fields each occupy two
decimal digits. This is an inherited representation; configurations must
respect its limits. In the published KS exports, `now_duration_ns` repeats the
profile's total segment duration and `solo_duration_ns` stores cumulative
duration through the current segment. The legacy names are unchanged; see
[the dataset guide](../data/profiling/README.md) for verified field semantics.
Fields ending in `_ns` are in nanoseconds, while scheduler timestamps and SLO
progress calculations use milliseconds. Preserve explicit unit conversions.

The default daemon's `scheduler()` selects a temporal window, `KS_L`. It does
not return an SM count. The client's spatial policy evaluates candidate SM
allocations, while the daemon also retains a `schedulerp()` spatial-policy
implementation from the research snapshot. Those similarly named functions
have different roles; their comments now describe the actual return values.

The default temporal policy computes SFR as actual progress divided by elapsed
SLO fraction, and PRR as actual progress divided by expected standalone
progress. It shapes these ratios, combines them with the configured risk
weight, and rounds/clamps the resulting segment window. See `scheduler.c` for
the exact equations and `STARS_W_SFR` / `STARS_RISK_DELTA` handling.

## Models and calibration

`src/modeling/interference.c` contains the original empirical interference
formula and SM-scaling exponent. Moving these functions into the modeling
directory did not refit their coefficients. The interference estimate retains
the original calibration's units; its coefficients alone do not establish a
portable time unit or accuracy guarantee for another GPU.

`lrmodel.c` contains a separate multivariate regression implementation. Its
presence does not mean the released runtime automatically retrains the
interference model. Some spatial-policy formulas still contain hardware- and
workload-specific constants, including the original 26-SM assumption. The
independent NV/MPSE runner instead queries the visible device's SM count.

## Compatibility and release scope

The migration preserves the destination repository's `STARS_*` names and the
existing profile/message field names. For example, the legacy spelling
`solo_duratoin_ns` remains in the protocol to avoid mixing a schema rename
with source migration. Use a library, daemon, and CLI built from the same
checkout; compatibility with binaries from the original private source tree
is not promised.

Comments, diagnostics, and release documentation are in English. Translation
preserves C statements, identifiers, format specifiers, and escaped characters
in messages; diagnostic prose changes are intentional. Consumers that match
old diagnostic text should use structured CSV fields instead.

Retained TODOs and commented research alternatives describe unfinished or
historical work, not validated features. Host builds cover all six scheduler
targets, and host integration checks cover the default daemon's control
protocol. The successful NV/MPSE GPU checks documented in
[RUNNING.md](RUNNING.md) do not establish end-to-end inference correctness or
reproduce the paper's performance results. Original model weights, complete
profiling pipelines, full experiment datasets, and analysis scripts remain
outside this source release. Archived profiling CSVs and derived KS tables are
now available in [`data/profiling/`](../data/profiling/).
