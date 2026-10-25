# Baseline configurations

For BERT, GPT-2, ResNet-50, ShuffleNet V2, VGG-16, and ViT, use the CLI
configuration described in the [model guide](../src/profiling/workloads/README.md).
The JSON files in this directory configure only the separate vector-add checks.

`NV.json` runs two ordinary CUDA clients. `MPSE.json` runs two MPS clients,
each requesting a six-SM context. Run either using:

```bash
python3 src/profiling/baseline.py --config configs/MPSE.json
```

Commands are relative to the repository root. `--dry-run` validates a config
without GPU access. Fields:

| Field | Meaning |
| --- | --- |
| `mode` | `NV` or `MPSE` |
| `device` | Visible CUDA device ordinal |
| `clients` | Number of concurrent worker processes |
| `sm_count` | SM count per MPSE context; ignored for NV |
| `elements` | Number of float elements in each vector |
| `iterations` | Measured queries per client |
| `warmup` | Warmup queries before the shared start barrier |
| `launches_per_query` | Kernel launches followed by one context synchronization |
| `timeout_seconds` | Deadline covering client warmup and measurement |

Full experiment datasets and scenario configurations remain pending. The six
actual model scripts and their configurable NV/MPSE modes are included.
