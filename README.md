# STARS

STARS is a research project for efficient spatio-temporal GPU resource
sharing in concurrent edge AI inference workloads.

The project explores how profiling, interference modeling, kernel-level
execution control, and runtime scheduling can be combined to improve GPU
utilization while satisfying the latency requirements of heterogeneous
inference tasks.

> **Release status:** This repository is currently under active preparation.
> We are releasing the documentation and artifact structure first. Source code
> and reproducibility tools will be released incrementally.

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
