# Model workloads

These are the six actual model scripts migrated from `Myshare/evaluation/`.
NV and MPSE are **execution modes**, not model names. The model forward passes,
text-generation calls, input shapes, request timing boundaries, and research
branches come from the corresponding original scripts.

| Script | Model | Input per request | Default pretrained source |
| --- | --- | --- | --- |
| `mytbert.py` | BERT base, loaded with `BertModel` | Token IDs and attention mask, sequence length 64 | Configurable Hugging Face ID `bert-base-uncased` |
| `mytgpt.py` | GPT-2, loaded with `GPT2LMHeadModel` | Random 16-token prompts followed by `generate()` | Configurable Hugging Face ID `gpt2` |
| `mytresnet.py` | ResNet-50 | Float image tensors, 3 x 224 x 224 | torchvision `ResNet50_Weights.IMAGENET1K_V2` |
| `mytshufflenet.py` | ShuffleNet V2 x1.0 | Float image tensors, 3 x 320 x 320 | torchvision `ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1` |
| `mytvgg.py` | VGG-16 | Float image tensors, 3 x 224 x 224 | torchvision `VGG16_Weights.IMAGENET1K_V1` |
| `mytvit.py` | ViT base, patch size 16 | Float image tensors, 3 x 224 x 224 | Configurable Hugging Face ID `google/vit-base-patch16-224` |

The inputs are generated tensors, as in the original scripts. The inference
models and pretrained weights are real; these scripts do not evaluate dataset
accuracy. The original GPT script defines a 32-token constant but comments out
`max_new_tokens` in the actual calls. This migration preserves the calls, so
generation length follows the checkpoint/library defaults rather than claiming
that every request generates 32 tokens.

## Installation and checkpoints

Use Linux, a working NVIDIA CUDA driver, and a CUDA-enabled PyTorch build.
MPSE also needs `nvidia-cuda-mps-control` on `PATH` and SM affinity support.
The locally verified Python environment used Python 3.12.9, PyTorch 2.4.1 (CUDA 12.4),
torchvision 0.19.1, and transformers 4.49.0; dependency versions are recorded in
[`requirements.txt`](requirements.txt). From the repository root:

```bash
python -m pip install -r src/profiling/workloads/requirements.txt
python src/profiling/workloads/mytbert.py --help
```

`--model-path` selects an existing Hugging Face checkpoint directory or model
ID for BERT/GPT-2/ViT. For ResNet/ShuffleNet/VGG it selects a PyTorch state-dict
file. With no override, the sources listed above use the normal framework cache
and may download missing weights. Checkpoint files are not committed.

The original transformer scripts used local directories. ViT's local config
identifies `google/vit-base-patch16-224`; the BERT/GPT-2 directories do not
establish an exact upstream checkpoint revision. The default IDs above are
portable loading choices, not proof that they reproduce the local files byte
for byte. Use the original local checkpoints for strict comparisons. The BERT
snapshot declares a masked-language-model architecture but is loaded through
`BertModel`, preserving the original script's choice; framework notices about
unused or newly initialized heads should be interpreted accordingly.

## Run each model

The launcher accepts any of `bert`, `gpt`, `resnet`, `shufflenet`, `vgg`, or `vit`:

```bash
# Native CUDA, actual ResNet-50 inference:
PYTHON=python bash scripts/run_model.sh resnet NV \
  --model-path /path/to/resnet50-11ad3fa6.pth \
  --slo-ms 100 --batch-mode fixed -b 1 -q 10

# MPS with a six-SM context, actual BERT inference:
PYTHON=python bash scripts/run_model.sh bert MPSE \
  --model-path /path/to/bert_checkpoint \
  --slo-ms 100 --batch-mode fixed -b 1 -sm 6 -q 10

# The same CLI is available directly in each of the six scripts:
python src/profiling/workloads/mytvit.py -scen NV \
  --model-path /path/to/vit_checkpoint --slo-ms 100 -q 10
```

The launcher starts a private MPS controller for MPSE, waits for readiness,
and stops that controller after the worker exits. Its logs are saved under
`results/stars-model-mps.*`. It does not stop pre-existing MPS services or
change GPU compute mode. Direct MPSE invocations require an already running
MPS service with matching `CUDA_MPS_PIPE_DIRECTORY` settings. To run several
models concurrently under one MPS service, start that service explicitly and
launch the model scripts directly from separate processes.

NV uses a private inactive MPS pipe so it cannot join an inherited controller.
Run NV/MPSE without `LD_PRELOAD`; the launcher rejects inherited preloads in
the worker rather than silently mixing STARS interception with a baseline.
Both modes require CUDA; there is no CPU fallback.

## Profiles, SLOs, batches, and synchronization

Supply either a measured `--latency-csv` or an explicit fixed `--slo-ms`.
The example `--slo-ms 100` is a user-selected deadline, not a measured latency
or an experimentally calibrated SLO. With no profile, `ISO_Lat(ms)` is `nan`.
For profile-based runs:

```bash
PYTHON=python bash scripts/run_model.sh shufflenet MPSE \
  --latency-csv /path/to/shufflenet.csv --batch-mode paper -sm 6 -q 100
```

The table must contain `batch`, `SM(%)`, and `latency(ms)` columns. Despite its
legacy name, `SM(%)` is used as an SM **count** in these scripts. A missing
batch/SM combination is an error. In NV mode this argument is a profile lookup
key and CSV label, not an enforced GPU allocation. For MPSE, actual affinity
is queried and recorded separately in the metadata; hardware may round it.

`--batch-mode fixed1` preserves the source helper's default batch size of 1.
`--batch-mode fixed` uses `-b`; `--batch-mode paper` preserves the original
weighted distribution (GPT-2 uses 1/2/4/8; the other models use 1/2/4/8/16).
The request-loop SLO multipliers remain 1.5/2/4/10 with weights
0.25/0.5/0.2/0.05 when a fixed SLO is not supplied. The legacy `-c` initial
coefficient and unused `-r` argument remain accepted; they do not override
this per-request sampler.

Default warmup covers 1/2/4/8/16, or 1/2/4/8 for GPT-2. Use
`--warmup-batches 1` for a short execution check; this changes warmup and
must be reported in performance comparisons. The `Our` memory-measurement
branch also retains its separate initial warmup. All scripts keep the original
`torch.backends.cudnn.enabled = False` setting and request-arrival behavior.

With no `-j`, measurement begins immediately after warmup. To coordinate
several processes, pass the same `-j /path/to/shared.json` to each, then write
`{"message": 1}` after they have warmed up. Prefer writing a temporary file and
renaming it over the signal file. Polling has a 120-second default timeout,
configurable with `--start-timeout`; it no longer spins indefinitely.

`-o` names an output CSV file, despite the retained `--output_dir` option name.
The CSV keeps the original columns, including the `timetamp` spelling, for
analysis compatibility. Writes are locked across processes, and standalone
latency is looked up for each request's actual batch size. Each completed run
also writes `<output>.json` with arguments, framework/device information, and
observed MPSE context affinity. Use separate output files if you need a distinct
metadata file for each concurrent worker.

## Research modes and migration changes

The original `Our`, `MPSD`, and `TEST` functions remain in every model script.
They have not received the same end-to-end validation as NV/MPSE:

- `Our` requires `libstars.so`, the STARS scheduler, MPS, and compatible
  `CLIENT_DIR` kernel profiles. It retains the six-context pool
  [2,4,6,8,10,26], memory reports, and SLO metadata transfer convention.
- `MPSD` retains the ordinary-context workload branch; external MPS resource
  configuration is still required. Selecting the function alone does not
  configure an MPS percentage.
- `TEST` retains the original diagnostic workload branch.

Portability changes are shared in `workload_support.py`: configurable paths,
typed/checkable CUDA calls, bounded optional start signaling, validated CLI
choices, and no silent CPU fallback. Vision model loading uses the installed
torchvision constructors instead of an old downloaded Git checkout. Unused
image-loader calls were removed because the original loops consume generated
tensors. `Profiler.py` preserves the original timing behavior; `utils.py` and
`savelat.py` carry the used batch/SLO and CSV functionality without unused
spreadsheet or image-loader dependencies. Errors loading weights now fail the
process instead of returning a successful exit status.

## Validation

On 2026-10-25, all six model scripts ran successfully in both NV and MPSE modes
on an RTX A2000 12GB with driver 550.107.02, using existing local pretrained
checkpoints. Each of the 12 runs used batch size 1, `--warmup-batches 1`, two
measured requests, and a fixed 100 ms SLO. Each produced two finite positive
latency rows; all six MPSE runs reported six effective SMs after inference.
This validates actual model execution, not inference accuracy or reproduction
of the paper's latency/SLO results. `make check-workloads` checks CLI usability,
profile lookup, CSV behavior, and synchronization without a GPU.
