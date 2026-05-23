#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j-test-gpu.out
#SBATCH --error=logs/%j-test-gpu.err

set -euo pipefail

echo "=== Node ==="
hostname

echo ""
echo "=== GRES allocated ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

echo ""
echo "=== GPU device ==="
ls -la /dev/nvhost-gpu

echo ""
echo "=== Tegra stats ==="
if command -v tegrastats &>/dev/null; then
    timeout 3 tegrastats || true
fi

echo ""
echo "=== nvidia-smi ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available (expected on Jetson)"
fi

echo ""
echo "=== Python GPU test ==="
python3 - <<'EOF'
import subprocess, sys

# Check for torch with CUDA
try:
    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name     : {torch.cuda.get_device_name(0)}")
        x = torch.ones(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print(f"Matrix multiply : OK  (result sum={y.sum().item():.0f})")
    else:
        print("CUDA not available to PyTorch — check JetPack install")
except ImportError:
    print("PyTorch not installed — skipping GPU compute test")

# Check for onnxruntime CUDA provider
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"\nONNX Runtime providers : {providers}")
    active = 'CUDAExecutionProvider' in providers
    print(f"CUDA provider available : {active}")
except ImportError:
    print("onnxruntime not installed — skipping")
EOF

echo ""
echo "=== Done ==="
