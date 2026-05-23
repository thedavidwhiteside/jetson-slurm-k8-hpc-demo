#!/bin/bash
#SBATCH --job-name=mlperf-resnet50
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j-resnet50.out
#SBATCH --error=logs/%j-resnet50.err

set -euo pipefail

MLPERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="$MLPERF_DIR/mlperf.sif"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"

mkdir -p "$MLPERF_DIR/logs"

apptainer exec --nv \
  --bind "$MLPERF_DIR:/mlperf" \
  "$SIF" \
  python3 /mlperf/src/run_inference.py \
    --model    /mlperf/models/resnet50_v1.onnx \
    --scenario SingleStream \
    --count    500 \
    --time     60

echo "Finished   : $(date)"
