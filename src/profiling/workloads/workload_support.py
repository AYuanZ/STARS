"""Portable configuration and CUDA support for the migrated model workloads."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time


class SmCount(C.Structure):
    _fields_ = [("val", C.c_uint)]


class AffinityValue(C.Union):
    _fields_ = [("smCount", SmCount)]


class Affinity(C.Structure):
    _fields_ = [("type", C.c_int), ("value", AffinityValue)]


class LazyCuda:
    """Load the driver on first use; declare pointer widths explicitly."""
    def __init__(self):
        self.library = None

    def __getattr__(self, name):
        if self.library is None:
            lib = C.CDLL("libcuda.so.1")
            pointer = C.c_void_p
            signatures = {
                "cuInit": [C.c_uint],
                "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
                "cuDeviceGetAttribute": [C.POINTER(C.c_int), C.c_int, C.c_int],
                "cuCtxCreate_v3": [C.POINTER(pointer), C.POINTER(Affinity), C.c_int, C.c_uint, C.c_int],
                "cuCtxSetCurrent": [pointer],
                "cuCtxGetCurrent": [C.POINTER(pointer)],
                "cuCtxGetExecAffinity": [C.POINTER(Affinity), C.c_int],
                "cuCtxSynchronize": [],
            }
            for symbol, args in signatures.items():
                function = getattr(lib, symbol)
                function.argtypes, function.restype = args, C.c_int
            self.library = lib
        return getattr(self.library, name)


libcuda = LazyCuda()
_contexts = []


def check_cuda(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")


def effective_sms():
    affinity = Affinity()
    check_cuda(libcuda.cuCtxGetExecAffinity(C.byref(affinity), 0), "cuCtxGetExecAffinity")
    return affinity.value.smCount.val


class CUDAContextManager:
    """Original fixed-SM context mechanism with checked, typed driver calls."""
    def create_custom_context(self, device_id, smcount, affinity_params=None):
        check_cuda(libcuda.cuInit(0), "cuInit")
        device = C.c_int()
        check_cuda(libcuda.cuDeviceGet(C.byref(device), device_id), "cuDeviceGet")
        total = C.c_int()
        check_cuda(libcuda.cuDeviceGetAttribute(C.byref(total), 16, device), "cuDeviceGetAttribute")
        if not 0 < smcount <= total.value:
            raise ValueError(f"SM count {smcount} is outside the device range [1, {total.value}]")
        affinity = Affinity(0, AffinityValue(SmCount(smcount)))
        context = C.c_void_p()
        check_cuda(libcuda.cuCtxCreate_v3(C.byref(context), C.byref(affinity), 1, 0, device), "cuCtxCreate_v3")
        actual = effective_sms()
        _contexts.append({"requested_sms": smcount, "effective_sms": actual, "context": context.value})
        print(f"CUDA affinity: requested={smcount}, effective={actual} SMs", flush=True)
        # Contexts must outlive tensors allocated by the original workloads.
        # The command-line worker exits after one workload, releasing them.
        return context

    def switch_to_custom_context(self, context):
        check_cuda(libcuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        return True


def get_model_path(name):
    defaults = {"bert": "bert-base-uncased", "gpt": "gpt2", "vit": "google/vit-base-patch16-224"}
    return os.environ.get("STARS_MODEL_PATH") or defaults[name]


def profile_path():
    return os.environ.get("STARS_LATENCY_CSV", "")


def load_vision_model(name):
    import torch
    from torchvision import models
    constructors = {"resnet": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
                    "shufflenet": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
                    "vgg": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1)}
    constructor, weights = constructors[name]
    path = os.environ.get("STARS_MODEL_PATH")
    if path:
        model = constructor(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
    # Use the installed torchvision API and its checkpoint cache, not a Git checkout.
    return constructor(weights=weights)


def warmup_batches(default):
    override = os.environ.get("STARS_WARMUP_BATCHES")
    return [int(value) for value in override.split(",")] if override else default


def wait_for_start(path):
    if not path:
        return
    deadline = time.monotonic() + float(os.environ.get("STARS_START_TIMEOUT", "120"))
    while time.monotonic() < deadline:
        try:
            with open(path) as file:
                if json.load(file)["message"] != 0:
                    return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for a nonzero message in {path}")


def parse_args(model):
    parser = argparse.ArgumentParser(description=f"Run the migrated {model} model workload.")
    parser.add_argument('-scen', '--scenario_type', choices=['NV', 'MPSE', 'MPSD', 'Our', 'TEST'], default='NV')
    parser.add_argument('-sm', '--sms', type=int, default=6, help='SM count, not a percentage')
    parser.add_argument('-q', '--query_num', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-m', '--model_name', default=model, help='Label written to the result CSV')
    parser.add_argument('-o', '--output_dir', default=f'results/{model}-{time.time_ns()}.csv', help='Output CSV file')
    parser.add_argument('-j', '--json_path', default='', help='Optional shared JSON start signal')
    parser.add_argument('-s', '--seed', type=int, default=22)
    parser.add_argument('-c', '--coefficient', type=float, default=2, help='Retained legacy initial-SLO coefficient')
    parser.add_argument('-r', '--random_slo', type=float, default=0, help='Retained legacy option; unused by the original request loops')
    parser.add_argument('--model-path', default=os.environ.get('STARS_MODEL_PATH', ''), help='HF directory/repository ID, or torchvision state_dict file')
    parser.add_argument('--latency-csv', default=os.environ.get('STARS_LATENCY_CSV', ''), help='Measured batch/SM latency profile')
    parser.add_argument('--slo-ms', type=float, default=None, help='Explicit fixed SLO; no profile table required')
    parser.add_argument('--batch-mode', choices=['fixed1', 'fixed', 'paper'], default=os.environ.get('STARS_BATCH_MODE', 'fixed1'))
    parser.add_argument('--warmup-batches', default='', help='Comma-separated warmup sizes; default preserves the original model list')
    parser.add_argument('--start-timeout', type=float, default=120)
    args = parser.parse_args()
    for key in ('sms', 'query_num', 'batch_size'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if not math.isfinite(args.start_timeout) or args.start_timeout <= 0:
        parser.error('--start-timeout must be positive and finite')
    if args.slo_ms is not None and (not math.isfinite(args.slo_ms) or args.slo_ms <= 0):
        parser.error('--slo-ms must be positive and finite')
    if not args.latency_csv and args.slo_ms is None:
        parser.error('provide --latency-csv or an explicit --slo-ms')
    if args.warmup_batches:
        try:
            if any(int(v) < 1 for v in args.warmup_batches.split(',')):
                raise ValueError()
        except ValueError:
            parser.error('--warmup-batches must contain positive integers')
    os.environ.update(STARS_MODEL_PATH=args.model_path, STARS_LATENCY_CSV=args.latency_csv,
                      STARS_BATCH_MODE=args.batch_mode, STARS_FIXED_BATCH=str(args.batch_size),
                      STARS_START_TIMEOUT=str(args.start_timeout), STARS_WARMUP_BATCHES=args.warmup_batches)
    if args.slo_ms is not None:
        os.environ['STARS_FIXED_SLO_MS'] = str(args.slo_ms)
    else:
        os.environ.pop('STARS_FIXED_SLO_MS', None)
    return args


def run_workload(namespace, args):
    import torch
    import numpy as np
    from utils import find_lat
    if args.scenario_type in ('NV', 'MPSE') and os.environ.get('LD_PRELOAD'):
        raise RuntimeError('Run NV/MPSE without LD_PRELOAD so the STARS hook does not change the baseline.')
    if args.scenario_type == 'Our' and 'libstars' not in os.environ.get('LD_PRELOAD', ''):
        raise RuntimeError('Our mode requires libstars.so, an active scheduler, MPS, and CLIENT_DIR kernel profiles.')
    # A private inactive pipe keeps NV clients from joining an existing MPS service.
    with tempfile.TemporaryDirectory(prefix='stars-model-') as temporary:
        if args.scenario_type == 'NV':
            os.environ['CUDA_MPS_PIPE_DIRECTORY'] = str(Path(temporary) / 'inactive')
        if not torch.cuda.is_available():
            raise RuntimeError('A working CUDA GPU is required; CPU fallback is disabled for these experiments.')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        output = Path(args.output_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        initial_batch = 1 if args.batch_mode == 'fixed1' else args.batch_size
        iso_lat = find_lat(args.sms, initial_batch, args.latency_csv) if args.latency_csv else float('nan')
        initial_slo = args.slo_ms if args.slo_ms is not None else iso_lat * args.coefficient
        print(f'Model={args.model_name}, mode={args.scenario_type}, requested SMs={args.sms}, queries={args.query_num}', flush=True)
        namespace[args.scenario_type](str(output), args.latency_csv, args.model_name, args.query_num,
                                     args.batch_size, args.sms, args.json_path, initial_slo, args.coefficient, iso_lat)
        torch.cuda.synchronize()
        actual = None
        if args.scenario_type == 'MPSE':
            actual = effective_sms()
            if actual != _contexts[-1]['effective_sms']:
                raise RuntimeError('The current CUDA context no longer has the requested MPSE affinity.')
        metadata = {**vars(args), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0), 'contexts': _contexts,
                    'device_sms': torch.cuda.get_device_properties(0).multi_processor_count,
                    'final_effective_sms': actual,
                    'measurement': 'actual model inference on generated input tensors'}
        Path(str(output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Completed; results: {output}', flush=True)
