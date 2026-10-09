#!/usr/bin/env bash
# ==============================================================================
# Blackwell Environment Bootstrap Script (RTX PRO 4500, B100, B200, sm_100/120)
# ==============================================================================

set -e

# Allow uv to operate seamlessly in system/conda environments
export UV_SYSTEM_PYTHON=1

echo "=== [1/5] Verifying GPU and System Prerequisites ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed or not in PATH."
    exit 1
fi

nvidia-smi

# Check CUDA Toolkit / NVCC (prefer CUDA 12.8 if installed)
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME="/usr/local/cuda-12"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif command -v nvcc &> /dev/null; then
    export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "Using CUDA_HOME: $CUDA_HOME"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" || true
    fi
fi

echo "=== [2/5] Installing uv, ninja, gdown, and modern wheel tooling ==="
pip install uv ninja gdown
uv pip install --upgrade setuptools wheel

echo "=== [3/5] Installing PyTorch with CUDA 12.8 (Blackwell SM_100/SM_120 Support) ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/5] Patching PyTorch cpp_extension to bypass NVCC version mismatch ==="
python -c '
import torch.utils.cpp_extension as ce
p = ce.__file__
with open(p, "r") as f:
    src = f.read()
target = "raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"
sub = "print(f\"[WARNING] CUDA version mismatch: detected {cuda_str_version} vs torch {torch.version.cuda}. Proceeding with build...\")"
if target in src:
    with open(p, "w") as f:
        f.write(src.replace(target, sub))
    print("[+] Patched PyTorch cpp_extension: CUDA version mismatch check bypassed.")
'

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0;10.0"

echo "=== [5/5] Running Hardware Diagnostic ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/check_blackwell_env.py"

echo "=================================================================="
echo " Blackwell environment is bootstrapped and verified!"
echo "=================================================================="
