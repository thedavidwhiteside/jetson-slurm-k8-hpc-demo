#!/usr/bin/env python3
"""
MLPerf Inference - ResNet50 image classification on Jetson.

Uses mlcommons loadgen when available, falls back to a standalone
timing loop that reports the same metrics (latency percentiles, QPS).

Synthetic data is used by default so no ImageNet registration is needed.
See README.md for swapping in real validation images.
"""

import argparse
import time
import numpy as np
import onnxruntime as ort

try:
    import mlperf_loadgen as lg
    HAS_LOADGEN = True
except ImportError:
    HAS_LOADGEN = False
    print("mlperf_loadgen not found — running in standalone timing mode\n")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_session(model_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    # Try CUDA first; fall back to CPU if the GPU provider isn't available.
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, opts, providers=providers)
    active = session.get_providers()[0]
    print(f"ONNX Runtime provider : {active}")
    return session


def synthetic_images(count):
    """Random ImageNet-shaped NCHW float32 batch."""
    return np.random.randn(count, 3, 224, 224).astype(np.float32)


def load_image(path):
    """Preprocess a single JPEG/PNG for ResNet50 (used with real data)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]          # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Standalone timing mode
# ---------------------------------------------------------------------------

def run_standalone(session, count, warmup):
    input_name = session.get_inputs()[0].name
    data = synthetic_images(1)

    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        session.run(None, {input_name: data})

    print(f"Benchmarking {count} samples (SingleStream)...")
    latencies = []
    t_start = time.perf_counter()
    for _ in range(count):
        t0 = time.perf_counter()
        session.run(None, {input_name: data})
        latencies.append(time.perf_counter() - t0)
    t_total = time.perf_counter() - t_start

    ms = np.array(latencies) * 1000
    print("\n=== Results ===")
    print(f"  Samples    : {count}")
    print(f"  Total time : {t_total:.2f} s")
    print(f"  Throughput : {count / t_total:.1f} QPS")
    print(f"  Latency")
    print(f"    mean     : {ms.mean():.2f} ms")
    print(f"    p50      : {np.percentile(ms, 50):.2f} ms")
    print(f"    p90      : {np.percentile(ms, 90):.2f} ms")
    print(f"    p99      : {np.percentile(ms, 99):.2f} ms")


# ---------------------------------------------------------------------------
# Official loadgen mode
# ---------------------------------------------------------------------------

def run_with_loadgen(session, scenario_str, count, time_limit_s):
    input_name = session.get_inputs()[0].name
    data = synthetic_images(count)

    def issue_queries(query_samples):
        for qs in query_samples:
            idx = qs.index % count
            session.run(None, {input_name: data[idx : idx + 1]})
            lg.QuerySamplesComplete([lg.QuerySampleResponse(qs.id, 0, 0)])

    def flush_queries():
        pass

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count, min(count, 500),
        lambda s: None,   # load_query_samples
        lambda s: None,   # unload_query_samples
    )

    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline":      lg.TestScenario.Offline,
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario_str]
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_duration_ms = time_limit_s * 1000
    settings.min_query_count = count

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False

    print(f"Scenario   : {scenario_str}")
    print(f"Min time   : {time_limit_s}s  |  Min samples : {count}\n")
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    print("\nSummary written to mlperf_log_summary.txt")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MLPerf ResNet50 inference on Jetson")
    ap.add_argument("--model",    required=True,  help="Path to resnet50_v1.onnx")
    ap.add_argument("--scenario", default="SingleStream", choices=["SingleStream", "Offline"])
    ap.add_argument("--count",    type=int, default=500,  help="Number of samples")
    ap.add_argument("--time",     type=int, default=60,   help="Min duration (seconds)", dest="time_limit")
    ap.add_argument("--warmup",   type=int, default=10,   help="Warmup iterations (standalone mode only)")
    args = ap.parse_args()

    session = build_session(args.model)

    if HAS_LOADGEN:
        run_with_loadgen(session, args.scenario, args.count, args.time_limit)
    else:
        run_standalone(session, args.count, args.warmup)


if __name__ == "__main__":
    main()
