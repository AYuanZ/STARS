"""VGG workload migrated from Myshare/evaluation/mytvgg.py.

Original workload author: AYuanzi. See README.md for portability changes,
model sources, and the distinction between model workloads and execution modes.
"""
from workload_support import (
    CUDAContextManager, libcuda, parse_args, run_workload,
    get_model_path, profile_path, load_vision_model, wait_for_start, warmup_batches,
)

# Parse --help before importing optional model packages or touching CUDA.
if __name__ == "__main__":
    _args = parse_args("vgg")

import torch
# sample execution (requires torchvision)
import time
import torchvision.models as models

from savelat import save_task_sm_lat_csv, save_sm_lat_tamp_csv
from Profiler import Profile, Profile2
import os
from datetime import datetime
import argparse

import ctypes
from ctypes import *
import json
import pandas as pd
import pynvml
import random
from utils import gen_rd_slo, gen_batch

torch.backends.cudnn.enabled = False  # Preserve the original cuDNN-disabled configuration


def get_gpu_memory_mib(device_id=0):
    """Read process, device, and PyTorch allocator memory usage in MiB."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        device_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        process_used_bytes = 0
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            used = process.usedGpuMemory
            if process.pid == os.getpid() and isinstance(used, int) and used >= 0:
                process_used_bytes += used

        mib = 1024 ** 2
        return {
            "process_used_mib": process_used_bytes / mib,
            "device_used_mib": device_memory.used / mib,
            "torch_allocated_mib": torch.cuda.memory_allocated(device_id) / mib,
            "torch_reserved_mib": torch.cuda.memory_reserved(device_id) / mib,
        }
    finally:
        pynvml.nvmlShutdown()


def print_context_memory_report(before, after_six, stage_label="After creating contexts [2,4,6,8,10,26]"):
    fixed_extra = after_six["process_used_mib"] - before["process_used_mib"]

    print("\n========== CUDA context memory measurement (NVML) ==========")
    print(
        "Primary context after initial warmup: "
        f"{before['process_used_mib']:.1f} MiB"
    )
    print(
        f"{stage_label}: "
        f"{after_six['process_used_mib']:.1f} MiB"
    )
    print(f"Increase relative to the initial warmup baseline: {fixed_extra:.1f} MiB")
    print(
        "Additional metric: PyTorch reserved "
        f"{before['torch_reserved_mib']:.1f} -> "
        f"{after_six['torch_reserved_mib']:.1f} MiB"
    )
    print(
        "Device memory (may include MPS server and other processes): "
        f"{before['device_used_mib']:.1f} -> "
        f"{after_six['device_used_mib']:.1f} MiB"
    )
    print("=================================================\n")


def print_three_stage_memory_report(before, after_contexts, after_batches):
    """Separate context creation from cumulative batch warmup memory costs."""
    print("\n========== Three-stage memory usage (MiB) ==========")
    print("Metric | memory_before_contexts | memory_after_six_contexts | memory_after_5_bs | context increment | warmup increment | total increment")
    for key in ["process_used_mib", "device_used_mib", "torch_allocated_mib", "torch_reserved_mib"]:
        initial, created, warmed = before[key], after_contexts[key], after_batches[key]
        print(f"{key} | {initial:.2f} | {created:.2f} | {warmed:.2f} | "
              f"{created-initial:.2f} | {warmed-created:.2f} | {warmed-initial:.2f}")
    print("The last two increments include batch expansion and runtime resources, not just context creation.\n")


def find_SLO(sms, batch_size):
    csv_file = profile_path()
    df = pd.read_csv(csv_file)

    # Look up latency by batch size and SM count
    result = df[(df['batch'] == batch_size) & (df['SM(%)'] == sms)]

    if not result.empty:
        latency = result['latency(ms)'].values[0]
        return int(round(latency))  # Round to the nearest integer
    else:
        raise ValueError("No latency profile entry matches this batch size and SM count.")
        return None

def generate_poisson_wait(num_events):
    lambda_rate = 10
    waiting_times = [round(random.expovariate(lambda_rate), 4) for _ in range(num_events)]
    return waiting_times

def gen_slo(initslo):
    slo_xs_values = [1.5, 2, 4, 10]
    slo_xs_weights = [0.2, 0.5, 0.2, 0.1]
    slo_xs = random.choices(slo_xs_values, weights=slo_xs_weights, k=1)[0]
    random_slo = int(round(slo * slo_xs))
    return random_slo

def Our(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):


    ctx_manager = CUDAContextManager()

    print(f"Requested SM count: {sms}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model("vgg")
    model.eval() # Evaluation mode
    if torch.cuda.is_available():
        model.to(device)

    # Warm up the primary context at batch size 16 to include model weights and inputs,
    # inference workspaces, and caches retained by PyTorch/CUDA after the first execution.
    with torch.no_grad():
        context_memory_warmup_input = torch.randn(16, 3, 224, 224, device=device)
        context_memory_warmup_output = model(context_memory_warmup_input)
    torch.cuda.synchronize()
    memory_before_contexts = get_gpu_memory_mib(0)

    # Create six fixed contexts with SM counts [2,4,6,8,10,26]
    custom_ctx1 = ctx_manager.create_custom_context(0, 2)
    custom_ctx2 = ctx_manager.create_custom_context(0, 4)
    custom_ctx3 = ctx_manager.create_custom_context(0, 6)
    custom_ctx4 = ctx_manager.create_custom_context(0, 8)
    custom_ctx5 = ctx_manager.create_custom_context(0, 10)
    custom_ctx6 = ctx_manager.create_custom_context(0, 26)
    libcuda.cuCtxSynchronize()
    memory_after_six_contexts = get_gpu_memory_mib(0)

    # Keep only six contexts; select the smallest candidate with SM count >= sms,
    # matching libstars find_context_by_sm_count() without creating duplicate contexts.
    context_pool = {
        2: custom_ctx1,
        4: custom_ctx2,
        6: custom_ctx3,
        8: custom_ctx4,
        10: custom_ctx5,
        26: custom_ctx6,
    }
    eligible_sm_counts = [sm_count for sm_count in context_pool if sm_count >= sms]
    if not eligible_sm_counts:
        raise ValueError(f"No candidate context can satisfy sms={sms}")
    selected_sm_count = min(eligible_sm_counts)
    custom_ctx = context_pool[selected_sm_count]

    print_context_memory_report(memory_before_contexts, memory_after_six_contexts)

    ctx_manager.switch_to_custom_context(custom_ctx)
    print(
        f"Created six candidate contexts; requested {sms} SMs, "
        f"reusing the {selected_sm_count}-SM context."
    )

    # Model warmup


    bs = warmup_batches([1, 2, 4, 8, 16])
    sm = [2, 4, 4, 6, 10]
    with torch.no_grad():
         for _, b in enumerate(bs):
            image = torch.randn(b, 3, 224, 224, device=device)
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities.cpu()

    sync_result = libcuda.cuCtxSynchronize()
    if sync_result != 0:
        raise RuntimeError(f"cuCtxSynchronize failed with error {sync_result}")
    memory_after_5_bs = get_gpu_memory_mib(0)
    print("&&&&&&&&&&&&&&&&&&&&&&&")
    print_three_stage_memory_report(
        memory_before_contexts, memory_after_six_contexts, memory_after_5_bs
    )
    print("************************")

    wait_for_start(json_path)


    # Run inference and measure latency

    start = time.time()
    dt = Profile(device=device)
    dt2 = Profile2(device=device)
    with torch.no_grad():
        for i in range(query_num):
            b= gen_batch(1)
            index = bs.index(b)
            # sms = sm[index]
            slo = gen_rd_slo(sms, b, lat_csv_path)
            # time.sleep(generate_poisson_wait(1)[0])

            with dt:
                with dt2:
                    image = torch.randn(b, 3, 224, 224, device=device)
                    slo2D =  slo * 100 + b
                    slo_t = torch.tensor([slo2D], dtype=torch.int32)
                    slo_t.to(device)
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities.cpu()

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)



        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))
    end = time.time()
    print("Total elapsed time (s): {}".format(end - start))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def NV(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model("vgg")
    model.eval() # Evaluation mode
    if torch.cuda.is_available():
        model.to(device)


    # Model warmup

    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            image = torch.randn(b, 3, 224, 224, device=device)
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities.cpu()

    print("Warmup complete.")

    wait_for_start(json_path)


    start = time.time()
    dt = Profile(device=device)
    dt2 = Profile2(device=device)
    with torch.no_grad():
        for i in range(query_num):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])
            print("i-------------", i)
            with dt:
                with dt2:
                    image = torch.randn(b, 3, 224, 224, device=device)
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities.cpu()

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))
    end = time.time()
    print("Total elapsed time (s): {}".format(end - start))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def MPSE(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):

    ctx_manager = CUDAContextManager()

    print(f"Requested SM count: {sms}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model("vgg")
    model.eval() # Evaluation mode
    if torch.cuda.is_available():
        model.to(device)


    # Create and switch to a custom context
    custom_ctx = ctx_manager.create_custom_context(0, sms)

    ctx_manager.switch_to_custom_context(custom_ctx)
    print(f"Switched to a custom context requesting {sms} SMs.")

    # Model warmup


    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            image = torch.randn(b, 3, 224, 224, device=device)
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities.cpu()

    wait_for_start(json_path)


    # Run inference and measure latency
    start = time.time()
    dt = Profile(device=device)
    dt2 = Profile2(device=device)
    with torch.no_grad():
        for i in range(query_num):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])
            print("i-------------", i)
            with dt:
                with dt2:
                    image = torch.randn(b, 3, 224, 224, device=device)
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities.cpu()

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))
    end = time.time()
    print("Total elapsed time (s): {}".format(end - start))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def MPSD(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model("vgg")
    model.eval() # Evaluation mode
    if torch.cuda.is_available():
        model.to(device)


    # Model warmup

    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            image = torch.randn(b, 3, 224, 224, device=device)
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities.cpu()

    wait_for_start(json_path)


    # Run inference and measure latency
    start = time.time()
    dt = Profile(device=device)
    dt2 = Profile2(device=device)
    with torch.no_grad():
        for i in range(query_num):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])
            print("i-------------", i)
            with dt:
                with dt2:
                    image = torch.randn(b, 3, 224, 224, device=device)
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities.cpu()

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))
    end = time.time()
    print("Total elapsed time (s): {}".format(end - start))

    # # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def TEST(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model("vgg")
    model.eval() # Evaluation mode
    if torch.cuda.is_available():
        model.to(device)


    # Model warmup

    bs = [1]
    with torch.no_grad():
         for _, b in enumerate(bs):
            image = torch.randn(b, 3, 224, 224, device=device)
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities.cpu()

    print("Warmup complete.")

    wait_for_start(json_path)


    start = time.time()
    dt = Profile(device=device)
    dt2 = Profile2(device=device)
    with torch.no_grad():
        for i in range(query_num):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            # time.sleep(generate_poisson_wait(1)[0])
            print("i-------------", i)
            with dt:
                with dt2:
                    image = torch.randn(b, 3, 224, 224, device=device)
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities.cpu()

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))
    end = time.time()
    print("Total elapsed time (s): {}".format(end - start))


if __name__ == "__main__":
    run_workload(globals(), _args)
