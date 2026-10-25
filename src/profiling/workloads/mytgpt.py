"""GPT workload migrated from Myshare/evaluation/mytgpt.py.

Original workload author: AYuanzi. See README.md for portability changes,
model sources, and the distinction between model workloads and execution modes.
"""
from workload_support import (
    CUDAContextManager, libcuda, parse_args, run_workload,
    get_model_path, profile_path, load_vision_model, wait_for_start, warmup_batches,
)

# Parse --help before importing optional model packages or touching CUDA.
if __name__ == "__main__":
    _args = parse_args("gpt")

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Language-model head for generation
from tqdm import tqdm
from Profiler import Profile, Profile2
import json
from savelat import save_sm_lat_tamp_csv
import os
from datetime import datetime
import argparse
import ctypes
from ctypes import *
import pandas as pd
import pynvml
import random
import time
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

    LOCAL_MODEL_PATH = get_model_path("gpt")
     # Text-generation parameters
    BATCH_SIZE = b
    PROMPT_LENGTH = 16             # Input prompt length
    NUM_TOKENS_TO_GENERATE = 32    # Legacy value; generation uses the checkpoint defaults below

    NUM_INFERENCES = query_num # Number of generation requests

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # Load the model and tokenizer from the configured path or repository identifier
    print(f"Loading model and tokenizer from '{LOCAL_MODEL_PATH}'...")
    try:
        # Load the text-generation model
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load model files from '{LOCAL_MODEL_PATH}'.")
        raise

    # GPT-2 has no padding token by default; use its end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    print("-" * 30)

    # Warm up the primary context at batch size 16 to include model weights and inputs,
    # inference workspaces, and caches retained by PyTorch/CUDA after the first execution.
    vocab_size = tokenizer.vocab_size
    with torch.no_grad():
        warmup_input_ids = torch.randint(0, vocab_size, (16, PROMPT_LENGTH), device=DEVICE)
        _ = model.generate(warmup_input_ids, pad_token_id=tokenizer.eos_token_id)
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
    print("-" * 30)

    # Prepare random-input generation parameters
    print(f"Prompt length: {PROMPT_LENGTH}; generation length follows the checkpoint configuration")
    print("-" * 30)

    # Text-generation evaluation
    # Warmup
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8])
    sm = [8, 8, 10, 12]
    with torch.no_grad():
         for _, b in enumerate(bs): # Warm up each configured batch size
            dummy_input_ids = torch.randint(0, vocab_size, (b, PROMPT_LENGTH), device=DEVICE)
            _ = model.generate(dummy_input_ids,
                            #    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                                pad_token_id=tokenizer.eos_token_id)

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

    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)    # Per-request measurement
    print(f"Running {NUM_INFERENCES} generation requests...")
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Generation"):
            # time.sleep(generate_poisson_wait(1)[0])
            BATCH_SIZE = gen_batch(2)
            index = bs.index(BATCH_SIZE)
            # sms = sm[index]
            slo = gen_rd_slo(sms, BATCH_SIZE, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])

            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LENGTH), device=DEVICE)
            with dt:
                with dt2:
                    slo2D =  slo * 100 + b
                    slo_t = torch.tensor([slo2D], dtype=torch.int32)
                    slo_t.to(DEVICE)
                    # Generate text with model.generate()
                    output = model.generate(
                        input_ids,
                        # max_new_tokens=NUM_TOKENS_TO_GENERATE,
                        pad_token_id=tokenizer.eos_token_id # Explicit padding token
                    )
            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, BATCH_SIZE, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)


def NV(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
    LOCAL_MODEL_PATH = get_model_path("gpt")
     # Text-generation parameters
    BATCH_SIZE = b
    PROMPT_LENGTH = 16             # Input prompt length
    NUM_TOKENS_TO_GENERATE = 32    # Legacy value; generation uses the checkpoint defaults below

    NUM_INFERENCES = query_num # Number of generation requests

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # Load the model and tokenizer from the configured path or repository identifier
    print(f"Loading model and tokenizer from '{LOCAL_MODEL_PATH}'...")
    try:
        # Load the text-generation model
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load model files from '{LOCAL_MODEL_PATH}'.")
        raise

    # GPT-2 has no padding token by default; use its end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    print("-" * 30)

    # Prepare random-input generation parameters
    vocab_size = tokenizer.vocab_size
    print(f"Prompt length: {PROMPT_LENGTH}; generation length follows the checkpoint configuration")
    print("-" * 30)

    # Text-generation evaluation
    # Warmup
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8])
    with torch.no_grad():
         for _, b in enumerate(bs): # Warm up each configured batch size
            dummy_input_ids = torch.randint(0, vocab_size, (b, PROMPT_LENGTH), device=DEVICE)
            _ = model.generate(dummy_input_ids,
                            #    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                                pad_token_id=tokenizer.eos_token_id)
    print("Warmup complete.")
    wait_for_start(json_path)

    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)    # Per-request measurement
    print(f"Running {NUM_INFERENCES} generation requests...")
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Generation"):
            # time.sleep(generate_poisson_wait(1)[0])
            BATCH_SIZE = gen_batch(2)
            slo = gen_rd_slo(sms, BATCH_SIZE, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])

            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LENGTH), device=DEVICE)
            with dt:
                with dt2:
                    # Generate text with model.generate()
                    output = model.generate(
                        input_ids,
                        # max_new_tokens=NUM_TOKENS_TO_GENERATE,
                        pad_token_id=tokenizer.eos_token_id # Explicit padding token
                    )
            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, BATCH_SIZE, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def MPSE(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
    ctx_manager = CUDAContextManager()

    LOCAL_MODEL_PATH = get_model_path("gpt")
     # Text-generation parameters
    BATCH_SIZE = b
    PROMPT_LENGTH = 16             # Input prompt length
    NUM_TOKENS_TO_GENERATE = 32    # Legacy value; generation uses the checkpoint defaults below

    NUM_INFERENCES = query_num # Number of generation requests

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # Load the model and tokenizer from the configured path or repository identifier
    print(f"Loading model and tokenizer from '{LOCAL_MODEL_PATH}'...")
    try:
        # Load the text-generation model
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load model files from '{LOCAL_MODEL_PATH}'.")
        raise

    # GPT-2 has no padding token by default; use its end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    print("-" * 30)

    # Create and switch to the custom context
    affinity_params = [{'type': 'CU_EXEC_AFFINITY_TYPE_SM_COUNT', 'param': {'smCount': {'val': sms}}}]
    custom_ctx = ctx_manager.create_custom_context(0, sms, affinity_params)

    ctx_manager.switch_to_custom_context(custom_ctx)
    print(f"Switched to a custom context requesting {sms} SMs.")
    print("-" * 30)


    # Prepare random-input generation parameters
    vocab_size = tokenizer.vocab_size
    print(f"Prompt length: {PROMPT_LENGTH}; generation length follows the checkpoint configuration")
    print("-" * 30)

    # Text-generation evaluation
    # Warmup
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8])
    with torch.no_grad():
         for _, b in enumerate(bs): # Warm up each configured batch size
            dummy_input_ids = torch.randint(0, vocab_size, (b, PROMPT_LENGTH), device=DEVICE)
            _ = model.generate(dummy_input_ids,
                            #    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                                pad_token_id=tokenizer.eos_token_id)
    print("Warmup complete.")
    wait_for_start(json_path)

    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)    # Per-request measurement
    print(f"Running {NUM_INFERENCES} generation requests...")
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Generation"):
            # time.sleep(generate_poisson_wait(1)[0])
            BATCH_SIZE = gen_batch(2)
            slo = gen_rd_slo(sms, BATCH_SIZE, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])

            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LENGTH), device=DEVICE)
            with dt:
                with dt2:
                    # Generate text with model.generate()
                    output = model.generate(
                        input_ids,
                        # max_new_tokens=NUM_TOKENS_TO_GENERATE,
                        pad_token_id=tokenizer.eos_token_id # Explicit padding token
                    )
            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, BATCH_SIZE, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)

def MPSD(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):

    LOCAL_MODEL_PATH = get_model_path("gpt")
     # Text-generation parameters
    BATCH_SIZE = b
    PROMPT_LENGTH = 16             # Input prompt length
    NUM_TOKENS_TO_GENERATE = 32    # Legacy value; generation uses the checkpoint defaults below

    NUM_INFERENCES = query_num # Number of generation requests

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # Load the model and tokenizer from the configured path or repository identifier
    print(f"Loading model and tokenizer from '{LOCAL_MODEL_PATH}'...")
    try:
        # Load the text-generation model
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load model files from '{LOCAL_MODEL_PATH}'.")
        raise

    # GPT-2 has no padding token by default; use its end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    print("-" * 30)

    # Prepare random-input generation parameters
    vocab_size = tokenizer.vocab_size
    print(f"Prompt length: {PROMPT_LENGTH}; generation length follows the checkpoint configuration")
    print("-" * 30)

    # Text-generation evaluation
    # Warmup
    print("Warming up...")
    bs = warmup_batches([1, 2, 4, 8])
    with torch.no_grad():
         for _, b in enumerate(bs): # Warm up each configured batch size
            dummy_input_ids = torch.randint(0, vocab_size, (b, PROMPT_LENGTH), device=DEVICE)
            _ = model.generate(dummy_input_ids,
                            #    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                                pad_token_id=tokenizer.eos_token_id)
    print("Warmup complete.")
    wait_for_start(json_path)

    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)    # Per-request measurement
    print(f"Running {NUM_INFERENCES} generation requests...")
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Generation"):
            # time.sleep(generate_poisson_wait(1)[0])
            BATCH_SIZE = gen_batch(2)
            slo = gen_rd_slo(sms, BATCH_SIZE, lat_csv_path)
            time.sleep(generate_poisson_wait(1)[0])

            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LENGTH), device=DEVICE)
            with dt:
                with dt2:
                    # Generate text with model.generate()
                    output = model.generate(
                        input_ids,
                        # max_new_tokens=NUM_TOKENS_TO_GENERATE,
                        pad_token_id=tokenizer.eos_token_id # Explicit padding token
                    )
            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, BATCH_SIZE, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))

    # save_sm_lat_tamp_csv(csv_file, model_name, b, sms, t, slo, time.time(), iso_lat)


def TEST(csv_file="", lat_csv_path="", model_name="", query_num=1000, b=4, sms=2, json_path="", initslo=0, c=0, iso_lat=0):
    LOCAL_MODEL_PATH = get_model_path("gpt")
     # Text-generation parameters
    BATCH_SIZE = b
    PROMPT_LENGTH = 16             # Input prompt length
    NUM_TOKENS_TO_GENERATE = 32    # Legacy value; generation uses the checkpoint defaults below

    NUM_INFERENCES = query_num # Number of generation requests

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # Load the model and tokenizer from the configured path or repository identifier
    print(f"Loading model and tokenizer from '{LOCAL_MODEL_PATH}'...")
    try:
        # Load the text-generation model
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
    except OSError as e:
        print(f"Cannot load model files from '{LOCAL_MODEL_PATH}'.")
        raise

    # GPT-2 has no padding token by default; use its end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    print("-" * 30)

    # Prepare random-input generation parameters
    vocab_size = tokenizer.vocab_size
    print(f"Prompt length: {PROMPT_LENGTH}; generation length follows the checkpoint configuration")
    print("-" * 30)

    # Text-generation evaluation
    # Warmup
    print("Warming up...")
    bs = [1]
    with torch.no_grad():
         for _, b in enumerate(bs): # Warm up each configured batch size
            dummy_input_ids = torch.randint(0, vocab_size, (b, PROMPT_LENGTH), device=DEVICE)
            _ = model.generate(dummy_input_ids,
                            #    max_new_tokens=NUM_TOKENS_TO_GENERATE,
                                pad_token_id=tokenizer.eos_token_id)
    print("Warmup complete.")
    wait_for_start(json_path)

    dt = Profile(device=DEVICE)
    dt2 = Profile2(device=DEVICE)    # Per-request measurement
    print(f"Running {NUM_INFERENCES} generation requests...")
    with torch.no_grad():
        for _ in tqdm(range(NUM_INFERENCES), desc="Generation"):
            # time.sleep(generate_poisson_wait(1)[0])
            BATCH_SIZE = gen_batch(1)
            slo = gen_rd_slo(sms, BATCH_SIZE, lat_csv_path)
            # time.sleep(generate_poisson_wait(1)[0])

            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LENGTH), device=DEVICE)
            with dt:
                with dt2:
                    # Generate text with model.generate()
                    output = model.generate(
                        input_ids,
                        # max_new_tokens=NUM_TOKENS_TO_GENERATE,
                        pad_token_id=tokenizer.eos_token_id # Explicit padding token
                    )
            t2 = dt2.dt * 1e3
            save_sm_lat_tamp_csv(csv_file, model_name, BATCH_SIZE, sms, t2, slo, time.time(), iso_lat)


        t = dt.t * 1e3 / query_num  # speeds per image
        print("Mean inference latency (ms): {}".format(t))


if __name__ == "__main__":
    run_workload(globals(), _args)
