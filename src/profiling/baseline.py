#!/usr/bin/env python3
"""Run concurrent native CUDA (NV) or fixed-SM MPS (MPSE) smoke workloads.

MPSE preserves the cuCtxCreate_v3 affinity mechanism used by the original
evaluation scripts. This synthetic workload does not reproduce inference results.
Only Python's standard library and the NVIDIA driver are needed.
"""
import argparse
import csv
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


class AffinityValue(C.Union):
    _fields_ = [("sm_count", C.c_uint)]


class Affinity(C.Structure):
    _fields_ = [("type", C.c_int), ("value", AffinityValue)]


class Driver:
    def __init__(self):
        try:
            self.lib = C.CDLL("libcuda.so.1")
        except OSError as exc:
            raise RuntimeError("NVIDIA driver library libcuda.so.1 is unavailable") from exc

    def call(self, name, types, *args):
        try:
            fn = getattr(self.lib, name)
        except AttributeError as exc:
            raise RuntimeError(f"CUDA driver does not export {name}") from exc
        fn.argtypes, fn.restype = types, C.c_int
        result = fn(*args)
        if result:
            message = C.c_char_p()
            error = self.lib.cuGetErrorString
            error.argtypes, error.restype = [C.c_int, C.POINTER(C.c_char_p)], C.c_int
            error(result, C.byref(message))
            detail = message.value.decode() if message.value else "unknown error"
            raise RuntimeError(f"{name}: CUDA error {result}: {detail}")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("mode") not in ("NV", "MPSE"):
        raise ValueError("mode must be NV or MPSE")
    for key in ("clients", "elements", "iterations", "launches_per_query", "sm_count", "timeout_seconds"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("device", "warmup"):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["elements"] > 2**28:
        raise ValueError("elements must be at most 2**28")
    return config


def worker(config, output, index):
    driver = Driver()
    call = driver.call
    ptr = C.c_void_p
    call("cuInit", [C.c_uint], 0)
    device, total_sms = C.c_int(), C.c_int()
    call("cuDeviceGet", [C.POINTER(C.c_int), C.c_int], C.byref(device), config["device"])
    # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16.
    call("cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int], C.byref(total_sms), 16, device)
    ctx = ptr()
    actual_sms = total_sms.value
    if config["mode"] == "MPSE":
        if config["sm_count"] > total_sms.value:
            raise ValueError(f"requested {config['sm_count']} SMs, device has {total_sms.value}")
        affinity = Affinity(0, AffinityValue(config["sm_count"]))
        call("cuCtxCreate_v3", [C.POINTER(ptr), C.POINTER(Affinity), C.c_int, C.c_uint, C.c_int],
             C.byref(ctx), C.byref(affinity), 1, 0, device)
        effective = Affinity()
        call("cuCtxGetExecAffinity", [C.POINTER(Affinity), C.c_int], C.byref(effective), 0)
        actual_sms = effective.value.sm_count
    else:
        call("cuCtxCreate_v2", [C.POINTER(ptr), C.c_uint, C.c_int], C.byref(ctx), 0, device)

    try:
        module, function = ptr(), ptr()
        ptx = Path(__file__).with_name("vector_add.ptx").read_bytes() + b"\0"
        call("cuModuleLoadData", [C.POINTER(ptr), C.c_char_p], C.byref(module), ptx)
        call("cuModuleGetFunction", [C.POINTER(ptr), ptr, C.c_char_p], C.byref(function), module, b"vector_add")
        count = config["elements"]
        buffers = [C.c_uint64() for _ in range(3)]
        for buffer in buffers:
            call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(buffer), count * 4)
        for buffer, bits in zip(buffers[:2], (0x3F800000, 0x40000000)):
            call("cuMemsetD32_v2", [C.c_uint64, C.c_uint, C.c_size_t], buffer, bits, count)
        n = C.c_uint(count)
        arguments = (ptr * 4)(*(C.cast(C.byref(v), ptr) for v in [*buffers, n]))

        def query():
            start = time.perf_counter_ns()
            for _ in range(config["launches_per_query"]):
                call("cuLaunchKernel", [ptr, C.c_uint, C.c_uint, C.c_uint,
                     C.c_uint, C.c_uint, C.c_uint, C.c_uint, ptr, C.POINTER(ptr), C.POINTER(ptr)],
                     function, (count + 255) // 256, 1, 1, 256, 1, 1, 0, None, arguments, None)
            call("cuCtxSynchronize", [])
            return (time.perf_counter_ns() - start) / 1e6

        for _ in range(config["warmup"]):
            query()
        (output / f"ready-{index}").touch()
        deadline = time.monotonic() + config["timeout_seconds"]
        while not (output / "start").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for the other clients")
            time.sleep(0.01)
        with (output / f"client-{index}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["mode", "client", "query", "requested_sms", "effective_sms", "latency_ms"])
            for iteration in range(config["iterations"]):
                elapsed = query()
                writer.writerow([config["mode"], index, iteration,
                                 config["sm_count"] if config["mode"] == "MPSE" else 0,
                                 actual_sms, elapsed])
        host = (C.c_float * count)()
        call("cuMemcpyDtoH_v2", [ptr, C.c_uint64, C.c_size_t], host, buffers[2], count * 4)
        if any(value != 3.0 for value in host):
            raise RuntimeError("GPU result verification failed (expected 1 + 2 = 3)")
        print(f"{config['mode']} client {index}: verified {count} elements; effective SMs={actual_sms}", flush=True)
    finally:
        # Destroying this context releases its allocations and modules, including on error.
        call("cuCtxDestroy_v2", [ptr], ctx)


def run(config, output):
    # Preflight before launching processes or creating any MPS service.
    Driver().call("cuInit", [C.c_uint], 0)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    processes, logs = [], []
    mps = None
    with tempfile.TemporaryDirectory(prefix="stars-mps-") as temporary:
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        env.pop("CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING", None)
        # An isolated pipe prevents NV clients from joining an existing MPS server.
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(Path(temporary) / "pipe")
        env["CUDA_MPS_LOG_DIRECTORY"] = str(output / "mps-logs")
        try:
            if config["mode"] == "MPSE":
                control = shutil.which("nvidia-cuda-mps-control")
                if control is None:
                    raise RuntimeError("nvidia-cuda-mps-control is not on PATH")
                Path(env["CUDA_MPS_PIPE_DIRECTORY"]).mkdir()
                Path(env["CUDA_MPS_LOG_DIRECTORY"]).mkdir()
                log = (output / "mps-control.log").open("w")
                logs.append(log)
                mps = subprocess.Popen([control, "-f"], env=env, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 10
                while True:
                    if mps.poll() is not None:
                        raise RuntimeError(f"MPS controller exited; see {output / 'mps-control.log'}")
                    probe = subprocess.run([control], input="get_server_list\n", text=True,
                                           env=env, capture_output=True, timeout=2)
                    if probe.returncode == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("MPS controller did not become ready")
                    time.sleep(0.1)
            for index in range(config["clients"]):
                log = (output / f"client-{index}.log").open("w")
                logs.append(log)
                processes.append(subprocess.Popen([
                    sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                    "--output", str(output), "--worker", str(index)], env=env,
                    stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + config["timeout_seconds"]
            while not all((output / f"ready-{i}").exists() for i in range(config["clients"])):
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError(f"a client exited during warmup; see {output}/client-*.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("clients did not become ready")
                time.sleep(0.05)
            (output / "start").touch()
            for process in processes:
                result = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                if result:
                    raise RuntimeError(f"a client failed; see {output}/client-*.log")
            print(f"{config['mode']}: all {len(processes)} clients passed; CSV files: {output}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if mps is not None:
                try:
                    subprocess.run([control], input="quit\n", text=True, env=env,
                                   capture_output=True, timeout=5)
                    mps.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mps.kill()
                    mps.wait()
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="new output directory (must not exist)")
    parser.add_argument("--dry-run", action="store_true", help="validate config without accessing CUDA")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        output = Path(args.output or f"results/{config['mode']}-{time.time_ns()}").resolve()
        if args.worker is None:
            run(config, output)
        else:
            worker(config, output, args.worker)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
        print(f"baseline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
