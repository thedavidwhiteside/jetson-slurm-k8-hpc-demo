# MLPerf Inference — ResNet50 on Jetson via Slurm

Small-scale MLPerf Inference benchmark running inside an Apptainer container,
submitted as a Slurm batch job to the Jetson compute node.

## What this runs

- **Benchmark**: MLPerf Inference — ResNet50 image classification
- **Scenario**: SingleStream (one sample at a time, measures latency)
- **Data**: Synthetic ImageNet-shaped tensors (no dataset registration needed)
- **Backend**: ONNX Runtime with CUDA via the NVIDIA L4T ML container
- **Harness**: mlcommons loadgen (falls back to a standalone timing loop if unavailable)

## Prerequisites

- Jetson provisioned with the Slurm playbook (`ansible-playbook jetson-slurm.yaml`)
- Apptainer installed on the Jetson (`apt install apptainer`)
- NVIDIA NGC credentials to pull the `nvcr.io/nvidia/l4t-ml` base image

### NGC login (one-time)

```bash
apptainer remote login --username '$oauthtoken' docker://nvcr.io
# paste your NGC API key when prompted
# get one free at https://ngc.nvidia.com
```

## Setup

Run once on the Jetson (or login node if `/mlperf` is on shared storage):

```bash
chmod +x mlperf/setup.sh
./mlperf/setup.sh
```

This will:
1. Build `mlperf/mlperf.sif` from `mlperf.def` (~2 GB, pulls from nvcr.io)
2. Download the ResNet50 ONNX model from Zenodo to `mlperf/models/`

## Running the benchmark

Submit via Slurm from the login node:

```bash
sbatch mlperf/jobs/resnet50-inference.sbatch
```

Monitor the job:

```bash
squeue                          # check job status
tail -f mlperf/logs/<jobid>-resnet50.out   # stream output
```

### Example output

```
Job ID     : 42
Node       : jetson1
Started    : Thu May 22 10:00:00 2025
ONNX Runtime provider : CUDAExecutionProvider

=== Results ===
  Samples    : 500
  Total time : 12.34 s
  Throughput : 40.5 QPS
  Latency
    mean     : 24.6 ms
    p50      : 24.1 ms
    p90      : 26.8 ms
    p99      : 31.2 ms
```

When mlcommons loadgen is available the summary is also written to
`mlperf_log_summary.txt` in the working directory.

## Changing the scenario

Edit the sbatch script or pass args directly for a quick test:

```bash
# Offline scenario — maximises throughput instead of measuring latency
apptainer exec --nv --bind "$PWD/mlperf:/mlperf" mlperf/mlperf.sif \
  python3 /mlperf/src/run_inference.py \
    --model /mlperf/models/resnet50_v1.onnx \
    --scenario Offline \
    --count 1000 \
    --time 60
```

## Using real ImageNet data

The default run uses synthetic random tensors — sufficient for measuring
infrastructure latency but not for accuracy validation.

To use real images:
1. Download the [ImageNet 2012 validation set](https://www.image-net.org/download) (requires free registration)
2. Place JPEG files in `mlperf/data/val/`
3. In `src/run_inference.py`, replace the `synthetic_images()` call with `load_image(path)` per sample

## Matching your JetPack version

The base image in `mlperf.def` is pinned to `r36.2.0-py3` (JetPack 6).
If your Jetson runs JetPack 5.x, change the `From:` line:

```
From: nvcr.io/nvidia/l4t-ml:r35.4.1-py3
```

Then rebuild: `rm mlperf/mlperf.sif && ./mlperf/setup.sh`

## Next steps

- **BERT** — add a second job script targeting the NLP benchmark
- **Multi-node** — use `#SBATCH --nodes=N` with MPI for distributed inference
- **TensorRT backend** — convert the ONNX model with `trtexec` inside the container for lower latency on Jetson
