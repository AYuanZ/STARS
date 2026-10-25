#!/usr/bin/env bash
# Run one actual model in NV or MPSE mode; manage only this run's MPS service.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: PYTHON=python3 bash scripts/run_model.sh {bert|gpt|resnet|shufflenet|vgg|vit} {NV|MPSE} [model arguments]" >&2
    exit 2
fi
model=$1
mode=$2
shift 2
case "$model" in bert|gpt|resnet|shufflenet|vgg|vit) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
case "$mode" in NV|MPSE) ;; *) echo "Use NV or MPSE; other research modes require manual integration." >&2; exit 2 ;; esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python_bin=${PYTHON:-python3}
if [[ "$mode" == NV ]]; then
    exec "$python_bin" "src/profiling/workloads/myt${model}.py" -scen NV "$@"
fi
command -v nvidia-cuda-mps-control >/dev/null
temporary=$(mktemp -d /tmp/stars-model-mps.XXXXXXXX)
controller_pid=""
cleanup() {
    if [[ -n "$controller_pid" ]]; then
        printf 'quit\n' | timeout 5s nvidia-cuda-mps-control >/dev/null 2>&1 || true
        if kill -0 "$controller_pid" 2>/dev/null; then
            kill "$controller_pid" 2>/dev/null || true
        fi
        wait "$controller_pid" 2>/dev/null || true
    fi
    rm -rf -- "$temporary"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_MPS_PIPE_DIRECTORY="$temporary/pipe"
export CUDA_MPS_LOG_DIRECTORY="$root/results/$(basename "$temporary")"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
nvidia-cuda-mps-control -f >"$CUDA_MPS_LOG_DIRECTORY/controller.log" 2>&1 &
controller_pid=$!
ready=0
for ((attempt=0; attempt<100; attempt++)); do
    if ! kill -0 "$controller_pid" 2>/dev/null; then
        echo "MPS controller exited; see $CUDA_MPS_LOG_DIRECTORY/controller.log" >&2
        exit 1
    fi
    if printf 'get_server_list\n' | timeout 2s nvidia-cuda-mps-control >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 0.1
done
if [[ "$ready" != 1 ]]; then
    echo "MPS controller did not become ready." >&2
    exit 1
fi
"$python_bin" "src/profiling/workloads/myt${model}.py" -scen MPSE "$@"
