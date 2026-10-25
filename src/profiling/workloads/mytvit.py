"""VIT workload migrated from Myshare/evaluation/mytvit.py.

Original workload author: AYuanzi. See README.md for portability changes,
model sources, and the distinction between model workloads and execution modes.
"""
from workload_support import (
    CUDAContextManager, libcuda, parse_args, run_workload,
    get_model_path, profile_path, load_vision_model, wait_for_start, warmup_batches,
)

# Parse --help before importing optional model packages or touching CUDA.
if __name__ == "__main__":
    _args = parse_args("vit")

import torch
from transformers import ViTForImageClassification
import time
from tqdm import tqdm
from Profiler import Profile, Profile2
import json
from savelat import save_sm_lat_tamp_csv
import argparse
import ctypes
from ctypes import *
import os
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
        return latency  # Preserve the measured value
    else:
        raise ValueError("No latency profile entry matches this batch size and SM count.")
        return None

def generate_poisson_wait(num_events):
    lambda_rate = 10
    waiting_times = [round(random.expovariate(lambda_rate), 4) for _ in range(num_events)]
    return waiting_times

def gen_slo(slo):
    slo_xs_values = [1.5, 2, 4]
    slo_xs_weights = [0.25, 0.5, 0.25]
    slo_xs = random.choices(slo_xs_values, weights=slo_xs_weights, k=1)[0]
    random_slo = int(round(slo * slo_xs))
    return random_slo

def Our(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):

    ctx_manager = CUDAContextManager()

           # Model configuration
    # Configured ViT checkpoint path or repository identifier
    LOCAL_MODEL_PATH = get_model_path("vit")

    # Input tensor shape for the ViT image workload
    BATCH_SIZE = b          # Initial batch size
    NUM_CHANNELS = 3        # RGB channels
    HEIGHT = 224            # Standard input height for vit-base-patch16-224
    WIDTH = 224             # Standard input width for vit-base-patch16-224

    # Evaluation request count
    NUM_INFERENCES = query_num   # Use a larger count for performance studies

    # The command-line entry point requires CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configured model
    try:
        # Only the classification model is needed for these tensor inputs
        model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load ViT model files from '{LOCAL_MODEL_PATH}'.")
        print(f"Details: {e}")
        raise

    # Move the model to the selected device
    model.to(DEVICE)
    # Evaluation mode disables dropout
    model.eval()

    # Warm up the primary context at batch size 16 to include model weights and inputs,
    # inference workspaces, and caches retained by PyTorch/CUDA after the first execution.
    with torch.no_grad():
        warmup_input_shape = (16, NUM_CHANNELS, HEIGHT, WIDTH)
        warmup_pixel_values = torch.randn(warmup_input_shape, device=DEVICE)
        _ = model(pixel_values=warmup_pixel_values)
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


    # Run inference and measure latency
    # Warm up inference before recording measurements
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8, 16])
    sm = [2, 4, 4, 6, 10]
    with torch.no_grad():
         for _, b in enumerate(bs):
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            # Generate a random tensor with the model's input shape
            # Use standard-normal image tensors, following the original workload
            dummy_pixel_values = torch.randn(input_shape, device=DEVICE)
            _ = model(pixel_values=dummy_pixel_values)

    sync_result = libcuda.cuCtxSynchronize()
    if sync_result != 0:
        raise RuntimeError(f"cuCtxSynchronize failed with error {sync_result}")
    memory_after_5_bs = get_gpu_memory_mib(0)
    print("&&&&&&&&&&&&&&&&&&&&&&&")
    print_three_stage_memory_report(
        memory_before_contexts, memory_after_six_contexts, memory_after_5_bs
    )
    print("************************")
    print("Warmup complete.")
    wait_for_start(json_path)

    # Measured inference loop
    print(f"Running {NUM_INFERENCES} inference requests...")
    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Inference"):
            b = gen_batch(1)
            index = bs.index(b)
            # sms = sm[index]
            slo = gen_rd_slo(sms, b, lat_csv_path)
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            time.sleep(generate_poisson_wait(1)[0])
            # 1. Generate a batch of synthetic image tensors
            pixel_values = torch.randn(input_shape, device=DEVICE)
            # 3. Run the model forward pass
            # This model expects the pixel_values argument
            with dt:
                with dt2:
                    slo2D =  slo * 100 + b
                    slo_t = torch.tensor([slo2D], dtype=torch.int32)
                    slo_t.to(DEVICE)
                    output = model(pixel_values=pixel_values)

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)

        t = dt.t * 1e3 / query_num  # speeds per image
        # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)
        print("Mean inference latency (ms): {}".format(t))

def NV(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
       # Model configuration
    # Configured ViT checkpoint path or repository identifier
    LOCAL_MODEL_PATH = get_model_path("vit")

    # Input tensor shape for the ViT image workload
    BATCH_SIZE = b          # Initial batch size
    NUM_CHANNELS = 3        # RGB channels
    HEIGHT = 224            # Standard input height for vit-base-patch16-224
    WIDTH = 224             # Standard input width for vit-base-patch16-224

    # Evaluation request count
    NUM_INFERENCES = query_num   # Use a larger count for performance studies

    # The command-line entry point requires CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configured model
    try:
        # Only the classification model is needed for these tensor inputs
        model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load ViT model files from '{LOCAL_MODEL_PATH}'.")
        print(f"Details: {e}")
        raise

    # Move the model to the selected device
    model.to(DEVICE)
    # Evaluation mode disables dropout
    model.eval()


    # Run inference and measure latency
    # Warm up inference before recording measurements
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            # Generate a random tensor with the model's input shape
            # Use standard-normal image tensors, following the original workload
            dummy_pixel_values = torch.randn(input_shape, device=DEVICE)
            _ = model(pixel_values=dummy_pixel_values)
    print("Warmup complete.")
    wait_for_start(json_path)

    # Measured inference loop
    print(f"Running {NUM_INFERENCES} inference requests...")
    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Inference"):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            time.sleep(generate_poisson_wait(1)[0])
            # 1. Generate a batch of synthetic image tensors
            pixel_values = torch.randn(input_shape, device=DEVICE)
            # 3. Run the model forward pass
            # This model expects the pixel_values argument
            with dt:
                with dt2:
                    output = model(pixel_values=pixel_values)

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)

        t = dt.t * 1e3 / query_num  # speeds per image
        # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)
        print("Mean inference latency (ms): {}".format(t))

def MPSE(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
    ctx_manager = CUDAContextManager()

           # Model configuration
    # Configured ViT checkpoint path or repository identifier
    LOCAL_MODEL_PATH = get_model_path("vit")

    # Input tensor shape for the ViT image workload
    BATCH_SIZE = b          # Initial batch size
    NUM_CHANNELS = 3        # RGB channels
    HEIGHT = 224            # Standard input height for vit-base-patch16-224
    WIDTH = 224             # Standard input width for vit-base-patch16-224

    # Evaluation request count
    NUM_INFERENCES = query_num   # Use a larger count for performance studies

    # The command-line entry point requires CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configured model
    try:
        # Only the classification model is needed for these tensor inputs
        model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load ViT model files from '{LOCAL_MODEL_PATH}'.")
        print(f"Details: {e}")
        raise

    # Move the model to the selected device
    model.to(DEVICE)
    # Evaluation mode disables dropout
    model.eval()

    # Create and switch to a custom context
    custom_ctx = ctx_manager.create_custom_context(0, sms)

    ctx_manager.switch_to_custom_context(custom_ctx)
    print(f"Switched to a custom context requesting {sms} SMs.")


    # Run inference and measure latency
    # Warm up inference before recording measurements
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            # Generate a random tensor with the model's input shape
            # Use standard-normal image tensors, following the original workload
            dummy_pixel_values = torch.randn(input_shape, device=DEVICE)
            _ = model(pixel_values=dummy_pixel_values)
    print("Warmup complete.")
    wait_for_start(json_path)

    # Measured inference loop
    print(f"Running {NUM_INFERENCES} inference requests...")
    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Inference"):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            time.sleep(generate_poisson_wait(1)[0])
            # 1. Generate a batch of synthetic image tensors
            pixel_values = torch.randn(input_shape, device=DEVICE)
            # 3. Run the model forward pass
            # This model expects the pixel_values argument
            with dt:
                with dt2:
                    output = model(pixel_values=pixel_values)

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)

        t = dt.t * 1e3 / query_num  # speeds per image

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)
    print("Mean inference latency (ms): {}".format(t))

def MPSD(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
       # Model configuration
    # Configured ViT checkpoint path or repository identifier
    LOCAL_MODEL_PATH = get_model_path("vit")

    # Input tensor shape for the ViT image workload
    BATCH_SIZE = b          # Initial batch size
    NUM_CHANNELS = 3        # RGB channels
    HEIGHT = 224            # Standard input height for vit-base-patch16-224
    WIDTH = 224             # Standard input width for vit-base-patch16-224

    # Evaluation request count
    NUM_INFERENCES = query_num   # Use a larger count for performance studies

    # The command-line entry point requires CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configured model
    try:
        # Only the classification model is needed for these tensor inputs
        model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load ViT model files from '{LOCAL_MODEL_PATH}'.")
        print(f"Details: {e}")
        raise

    # Move the model to the selected device
    model.to(DEVICE)
    # Evaluation mode disables dropout
    model.eval()

    # Prepare random-input generation parameters



    # Run inference and measure latency
    # Warm up inference before recording measurements
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8, 16])
    with torch.no_grad():
         for _, b in enumerate(bs):
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            # Generate a random tensor with the model's input shape
            # Use standard-normal image tensors, following the original workload
            dummy_pixel_values = torch.randn(input_shape, device=DEVICE)
            _ = model(pixel_values=dummy_pixel_values)
    print("Warmup complete.")
    wait_for_start(json_path)

    # Measured inference loop
    print(f"Running {NUM_INFERENCES} inference requests...")
    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Inference"):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            time.sleep(generate_poisson_wait(1)[0])
            # 1. Generate a batch of synthetic image tensors
            pixel_values = torch.randn(input_shape, device=DEVICE)
            # 3. Run the model forward pass
            # This model expects the pixel_values argument
            with dt:
                with dt2:
                    output = model(pixel_values=pixel_values)

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)

        t = dt.t * 1e3 / query_num  # speeds per image
        # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)
        print("Mean inference latency (ms): {}".format(t))

def TEST(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
       # Model configuration
    # Configured ViT checkpoint path or repository identifier
    LOCAL_MODEL_PATH = get_model_path("vit")

    # Input tensor shape for the ViT image workload
    BATCH_SIZE = b          # Initial batch size
    NUM_CHANNELS = 3        # RGB channels
    HEIGHT = 224            # Standard input height for vit-base-patch16-224
    WIDTH = 224             # Standard input width for vit-base-patch16-224

    # Evaluation request count
    NUM_INFERENCES = query_num   # Use a larger count for performance studies

    # The command-line entry point requires CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configured model
    try:
        # Only the classification model is needed for these tensor inputs
        model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load ViT model files from '{LOCAL_MODEL_PATH}'.")
        print(f"Details: {e}")
        raise

    # Move the model to the selected device
    model.to(DEVICE)
    # Evaluation mode disables dropout
    model.eval()


    # Run inference and measure latency
    # Warm up inference before recording measurements
    print("Warming up...")
    bs = [1]
    with torch.no_grad():
         for _, b in enumerate(bs):
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)
            # Generate a random tensor with the model's input shape
            # Use standard-normal image tensors, following the original workload
            dummy_pixel_values = torch.randn(input_shape, device=DEVICE)
            _ = model(pixel_values=dummy_pixel_values)
    print("Warmup complete.")
    wait_for_start(json_path)

    # Measured inference loop
    print(f"Running {NUM_INFERENCES} inference requests...")
    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Inference"):
            b= gen_batch(1)
            slo = gen_rd_slo(sms, b, lat_csv_path)
            # time.sleep(generate_poisson_wait(1)[0])
            input_shape = (b, NUM_CHANNELS, HEIGHT, WIDTH)

            # 1. Generate a batch of synthetic image tensors
            pixel_values = torch.randn(input_shape, device=DEVICE)
            # 3. Run the model forward pass
            # This model expects the pixel_values argument
            with dt:
                with dt2:
                    output = model(pixel_values=pixel_values)

            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t2, slo, time.time(), iso_lat)

        t = dt.t * 1e3 / query_num  # speeds per image
        # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)
        print("Mean inference latency (ms): {}".format(t))


if __name__ == "__main__":
    run_workload(globals(), _args)
