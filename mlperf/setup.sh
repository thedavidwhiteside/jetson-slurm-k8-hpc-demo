#!/bin/bash
# Build the MLPerf Apptainer image and download the ResNet50 model.
# Run once on the Jetson (or from the login node if storage is shared).
#
# Requires:
#   - apptainer installed
#   - NVIDIA NGC credentials (for nvcr.io base image)
#       docker login nvcr.io  OR  apptainer remote login --username '$oauthtoken' docker://nvcr.io
#   - ~3 GB free disk space for the .sif image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
LOG_DIR="$SCRIPT_DIR/logs"
SIF="$SCRIPT_DIR/mlperf.sif"

mkdir -p "$MODEL_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Build Apptainer image
# ---------------------------------------------------------------------------

if [[ -f "$SIF" ]]; then
  echo "==> Image already exists: $SIF"
  echo "    Delete it and re-run to rebuild."
else
  echo "==> Building Apptainer image (this pulls ~2 GB from nvcr.io)..."
  apptainer build "$SIF" "$SCRIPT_DIR/mlperf.def"
  echo "==> Image built: $SIF"
fi

# ---------------------------------------------------------------------------
# Download ResNet50 ONNX model (Zenodo, no login required)
# ---------------------------------------------------------------------------

MODEL_PATH="$MODEL_DIR/resnet50_v1.onnx"
MODEL_URL="https://zenodo.org/record/4735647/files/resnet50_v1.onnx"

if [[ -f "$MODEL_PATH" ]]; then
  echo "==> Model already present: $MODEL_PATH"
else
  echo "==> Downloading ResNet50 ONNX model (~100 MB)..."
  wget -q --show-progress -O "$MODEL_PATH" "$MODEL_URL"
  echo "==> Model saved: $MODEL_PATH"
fi

echo ""
echo "Setup complete. Submit the benchmark with:"
echo ""
echo "  sbatch $SCRIPT_DIR/jobs/resnet50-inference.sbatch"
