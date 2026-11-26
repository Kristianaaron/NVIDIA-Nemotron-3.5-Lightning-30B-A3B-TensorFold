#!/bin/bash
# Build the TensorFold native engine and download the quantized weights.
set -euo pipefail

ENGINE_REPO="${ENGINE_REPO:-https://github.com/Kristianaaron/TensorFold.git}"
ENGINE_REF="${ENGINE_REF:-c6b6d502f18c9da1ffc335561735231ad0e5cb09}"
WEIGHTS_REPO="${WEIGHTS_REPO:-Krishando/Nemotron-3.5-Lightning-30B-A3B-EXL3-2.3bpw}"
WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
SM="${SM:-120}"   # 120 = RTX 50-series (Blackwell consumer)

echo "=== GPU ==="
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found; install the NVIDIA driver."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== CUDA 13 ==="
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc (set CUDA_HOME)."; exit 1; }
"$CUDA_HOME/bin/nvcc" --version | tail -1

echo "=== Zig 0.17.0 ==="
command -v zig >/dev/null || { echo "ERROR: zig not found; install Zig 0.17.0 from https://ziglang.org/download/"; exit 1; }
zv=$(zig version); [ "$zv" = "0.17.0" ] || echo "WARNING: tested with Zig 0.17.0, found $zv"

echo "=== Engine ==="
[ -d tensorfold ] || git clone --filter=blob:none --branch nemotron-lightning-5070ti "$ENGINE_REPO" tensorfold
git -C tensorfold fetch -q origin nemotron-lightning-5070ti
git -C tensorfold checkout -q "$ENGINE_REF"
(cd tensorfold && zig build native -Doptimize=ReleaseFast -Dsm="$SM" -Dnvcc="$CUDA_HOME/bin/nvcc")
ls -la tensorfold/zig-out/native/bin/tensorfold-native

echo "=== Python tools (download + bench) ==="
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate && pip install -q "huggingface_hub>=1.0" tokenizers

echo "=== Weights (~11.6 GB) ==="
hf download "$WEIGHTS_REPO" --local-dir "$WEIGHTS_DIR"

echo "Done. Start the server with ./start.sh"
