# Building and running this snapshot

## Requirements and verified environment

The runtime builds on Linux with a GNU C99 compiler, GNU Make, glibc, libdl,
libpthread, and libm. Minimal CUDA declarations are included; the shared library
resolves CUDA/NVML symbols at runtime. Building it does not need `nvcc`.
The NV/MPSE runner uses Python 3.8+ and its standard library, `libcuda.so.1`,
and an NVIDIA GPU. MPSE additionally requires the MPS control executable and
driver/GPU support for `cuCtxCreate_v3` with SM-count execution affinity.
An unsupported configuration fails rather than falling back to NV or CPU.

Verified locally on 2026-10-24:

| Component | Version / result |
| --- | --- |
| GCC | 8.5.0 |
| Python | 3.12.9 |
| NVIDIA driver | 550.107.02 |
| CUDA toolkit used for PTX assembly validation | 12.4 |
| GPU | NVIDIA RTX A2000 12GB, 26 SMs |
| Host build | Shared library, CLI, all six scheduler targets compiled |
| Host integration | CLI off/on/time-quantum commands processed by daemon |
| NV execution | Two clients, 50 queries each; GPU output verified; 26 SMs |
| MPSE execution | Two clients, 50 queries each; GPU output verified; 6 SMs each |

The GPU tests used `configs/NV.json` and `configs/MPSE.json` unchanged.
Both compute vector addition on 1,048,576 elements, with five warmup queries
and ten kernel launches per measured query. Timings include Python/driver
launch overhead and context synchronization. These are execution checks, not
inference performance measurements. GPU and Unix-socket tests require access
outside sandboxes that prohibit those devices or socket operations.

## NV and MPSE

From the repository root:

```bash
python3 src/profiling/baseline.py --config configs/NV.json --dry-run
python3 src/profiling/baseline.py --config configs/NV.json
python3 src/profiling/baseline.py --config configs/MPSE.json
```

`--dry-run` validates the configuration without loading CUDA. `--output PATH`
selects a new output directory; existing directories are rejected to preserve
earlier results. Each successful run contains the resolved `config.json`,
`client-N.log`, and `client-N.csv`. MPSE also preserves controller logs.
Errors are nonzero exits; worker startup failures are described in the client
logs. All clients finish warmup before the parent releases their start barrier.

NV creates ordinary CUDA contexts and uses an isolated, inactive MPS pipe path.
MPSE starts a foreground controller in a temporary private pipe directory,
creates a context with a fixed SM affinity in each client, and queries the
actual SM count. Hardware may round the requested count. The runner does not
change GPU compute mode or stop any pre-existing MPS service. It removes
inherited `LD_PRELOAD` and MPS percentage/partition settings from its workers
so those do not alter the chosen baseline. `CUDA_VISIBLE_DEVICES` is preserved;
`device` is an ordinal in that visible device list.

The baseline workload is intentionally independent of model checkpoints,
datasets, the original external JSON barrier, and hardcoded home directories.
It validates the same native-context / fixed-SM-context distinction used by
the original NV/MPSE inference functions. Full ResNet/BERT/GPT/etc. experiments,
their weight provenance, and original experiment datasets are not included.

## STARS runtime and schedulers

```bash
make -j4
make schedulers
make check
# Select another temporal policy; a clean build is not necessary:
make SCHEDULER=EDF
# Restore the STARS scheduler:
make SCHEDULER=stars
```

Available policies: `stars`, `FIFO`, `EDF`, `EDF-noP`, `SJF`, and `SRTF`.
`build/scheduler-POLICY` preserves each built variant; `build/stars-scheduler`
is the currently selected one. `make -C src/runtime/core` forwards to the root
build. Generated binaries and local results are ignored by Git.

Start the selected scheduler and use the CLI in another terminal:

```bash
export STARS_SOCKET_DIR=/tmp/stars-demo
./build/stars-scheduler
# In another terminal, set the same STARS_SOCKET_DIR first:
./build/starsctl --anti-thrash=off
./build/starsctl --anti-thrash=on
./build/starsctl --set-tq=2
```

The directory's parent must exist. Use a different directory for each daemon;
the original socket-binding code replaces an existing socket at the chosen
path. Long options with values use `--name=value`. Stop the foreground daemon
with Ctrl-C. The CLI retains the original control interface; individual
research schedulers do not necessarily use the original time-quantum thread.

For the full STARS scheduling path, the application must use the original
workload metadata/context integration and compatible kernel profiles, as well
as MPS. This is not a claim that arbitrary CUDA programs become STARS workloads
just by preloading the library. Relevant variables:

| Variable | Purpose |
| --- | --- |
| `STARS_SOCKET_DIR` | Shared daemon/client socket directory; default `/tmp/stars` |
| `STARS_DEBUG` | Enables runtime debug logging when set |
| `STARS_LOG_PATH` | Runtime file log; default `/tmp/stars-predict.log` |
| `OUT_DIR` | Existing directory for interception CSVs; default `/tmp` |
| `CLIENT_DIR` | Kernel-profile path prefix, with trailing slash |
| `STARS_W_SFR` | Temporal risk weight in [0,1]; default 0.7 |
| `STARS_RISK_DELTA` | Risk deadband parameter in (0,1); default 0.2 |
| `STARS_BETA_SCHED` | Spatial utility weight in [0,1]; default 0.5 |
| `S` | Original workload-type selector used in spatial prediction |

The runtime preserves the destination repository's `STARS_*` environment and
symbol naming. Original `NVSHARE_*` settings are not aliases. The profile loader
expects `${CLIENT_DIR}${batch}/ks_part/ks_summary_sm_${sm}.csv` for batches
1, 2, 4, 8, and 16. Its eight CSV columns are:

```text
ks_id,kernel_num,duration_ns,sm_usage,dram_demand,l2_demand,now_duration_ns,solo_duration_ns
```

The predictor coefficients and several resource assumptions remain those of
the research snapshot; recalibrate before interpreting results on different
hardware. The inherited C sources still produce warnings for unused research
branches and related legacy code. Full STARS inference and baseline scheduling
behavior has not been validated end to end by these smoke tests.

## Source mapping and migration changes

The source is the working-tree snapshot under `Myshare/myshare/src`, including
its local changes, rather than only files in its latest Git commit. The source
directory was left unchanged. Existing STARS runtime files were retained.

| Source component | Destination |
| --- | --- |
| `client.*`, `xopt.*`, `utlist.h`, `snprintf.c`, symbol map | `src/runtime/core/` |
| `cinfo.*` | `src/profiling/` |
| `lrmodel.*` | `src/modeling/` |
| Empirical interference and SM scaling formulas from `scheduler.c` | `src/modeling/interference.*`, called by the default scheduler |
| `scheduler.c`, FIFO/EDF/EDF-noP/SJF/SRTF variants, `queuing.*` | `src/scheduler/` |
| `LICENSE` and source notices | Root license and retained file notices |

The constant-return `predict.c` placeholder, obsolete scheduler variants with
incompatible interfaces, backup copies, generated binaries, logs, Kubernetes
deployment files, and presentation documents were not migrated. The new
`baseline.py`, PTX workload, and configurations provide self-contained NV/MPSE
execution; the MPSE affinity mechanism follows the neighboring original
evaluation scripts.

Integration fixes include complete build/link dependencies, consistent STARS
symbols, configurable bounded socket paths, matching CLI/daemon message
layouts, initialized CLI options, consistent SM/batch array sizes, separate
FIFO queue functions for task and active-task layouts, safe CSV-open handling,
and corrected scheduler array-copy direction and disconnected-client lookup.
These changes do not replace the scheduling algorithms or fit new predictors.
