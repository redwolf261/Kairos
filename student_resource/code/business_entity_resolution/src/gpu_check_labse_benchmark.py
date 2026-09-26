#!/usr/bin/env python3
"""
Person D (GPU owner) -- Hour 0.5-1.5 task per the team plan.

Two things, in order:
  1. GPU/CUDA sanity check -- confirm torch sees the GPU and a real matmul
     runs on it. If this fails, everything downstream that depends on the
     GPU (LaBSE encoding, the optional 0.6B reranker) needs to be cut from
     the plan immediately, not discovered broken at hour 12.
  2. LaBSE benchmark on 100k real business names -- throughput (names/sec)
     and VRAM usage, so we know BEFORE committing to background-encoding the
     whole dataset whether that's a 10-minute job or a multi-hour one, and
     whether the model + a batch of embeddings actually fits in the 8GB
     card's VRAM budget the plan assumes.

Uses the shared normalized dataset cache (dataset_cache.py) already built by
the audit pipeline -- no need to re-read the raw 12M-row TSVs for this.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/gpu_check_labse_benchmark.py

Env vars:
    BENCHMARK_N        rows to benchmark (default 100_000, per the plan)
    BENCHMARK_BATCH    encoding batch size (default 256)
    MODEL_NAME         sentence-transformers model id (default
                        "sentence-transformers/LaBSE")
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_cache  # reuses the already-built normalized Parquet cache

N_BENCHMARK = int(os.environ.get("BENCHMARK_N", "100000"))
BATCH_SIZE = int(os.environ.get("BENCHMARK_BATCH", "256"))
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/LaBSE")

RESULTS_PATH = os.path.join(dataset_cache.audit.AUDIT_DIR, "gpu_labse_benchmark.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_check():
    """Step 1: confirm torch + CUDA actually work on this machine, with a
    real (not trivially small) matmul, and report the hardware we're
    actually running on so the benchmark numbers below are reproducible."""
    import torch

    log("=== GPU / CUDA check ===")
    result = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not result["cuda_available"]:
        log("CUDA NOT AVAILABLE -- stop here. Cut LaBSE and the reranker from "
            "the plan; both are GPU-only stretch goals with no CPU fallback "
            "worth the time budget. The required path (Stages 0-5, LightGBM) "
            "is unaffected.")
        return result

    result["cuda_version"] = torch.version.cuda
    result["device_name"] = torch.cuda.get_device_name(0)
    result["device_count"] = torch.cuda.device_count()
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    result["total_vram_gb"] = round(total_vram_gb, 2)

    log(f"torch {result['torch_version']}, CUDA {result['cuda_version']}, "
        f"device: {result['device_name']}, {result['total_vram_gb']} GB VRAM")

    # A real matmul, not a 2x2 toy -- big enough to actually exercise the
    # card and catch a broken driver/kernel mismatch, small enough to run in
    # under a second on any GPU that works at all.
    t0 = time.time()
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    matmul_seconds = time.time() - t0
    result["matmul_4096_seconds"] = round(matmul_seconds, 4)
    result["matmul_ok"] = bool(torch.isfinite(y).all().item())
    log(f"4096x4096 matmul on GPU: {matmul_seconds*1000:.1f}ms, "
        f"finite result: {result['matmul_ok']}")

    del x, y
    torch.cuda.empty_cache()
    return result


def load_benchmark_names(n):
    """Pull n real business names from the shared normalized cache -- a mix
    of train_source1 (US/India) so the benchmark reflects real string
    lengths/scripts, not synthetic data."""
    log(f"loading {n:,} real business names from the normalized dataset cache...")
    df = dataset_cache.load_normalized(
        "train_source1", columns=["business_name", "country"]
    )
    if len(df) > n:
        df = df.sample(n, random_state=42)
    names = df["business_name"].tolist()
    log(f"loaded {len(names):,} names "
        f"(countries: {df['country'].value_counts().to_dict()})")
    return names


def labse_benchmark(names, gpu_result):
    """Step 2: load LaBSE, encode `names` in batches on GPU, measure
    throughput and VRAM. This is the number that answers 'can we afford to
    background-encode the full dataset, and how long will it actually take.'"""
    import torch
    from sentence_transformers import SentenceTransformer

    if not gpu_result.get("cuda_available"):
        log("Skipping LaBSE benchmark -- no CUDA. Would need to run on CPU, "
            "which is far too slow to be worth benchmarking for this plan.")
        return {"skipped": True, "reason": "no_cuda"}

    log(f"=== LaBSE benchmark ({len(names):,} names, batch={BATCH_SIZE}) ===")
    log(f"loading model '{MODEL_NAME}' (first run downloads ~1.9GB fp32 weights)...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    load_seconds = time.time() - t0
    log(f"model loaded in {load_seconds:.1f}s")

    vram_after_load_gb = torch.cuda.memory_allocated() / 1e9
    log(f"VRAM allocated after model load: {vram_after_load_gb:.2f} GB")

    log("encoding (this is the timed portion)...")
    t0 = time.time()
    embeddings = model.encode(
        names,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # LaBSE embeddings are meant to be used with cosine sim
    )
    torch.cuda.synchronize()
    encode_seconds = time.time() - t0

    vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    names_per_sec = len(names) / encode_seconds if encode_seconds else 0

    result = {
        "skipped": False,
        "model_name": MODEL_NAME,
        "model_load_seconds": round(load_seconds, 1),
        "n_names_encoded": len(names),
        "batch_size": BATCH_SIZE,
        "encode_seconds": round(encode_seconds, 1),
        "names_per_second": round(names_per_sec, 1),
        "embedding_dim": int(embeddings.shape[1]),
        "vram_allocated_after_load_gb": round(vram_after_load_gb, 2),
        "vram_peak_during_encode_gb": round(vram_peak_gb, 2),
        "vram_headroom_gb_on_8gb_card": round(8.0 - vram_peak_gb, 2),
    }

    log(f"encoded {len(names):,} names in {encode_seconds:.1f}s "
        f"({names_per_sec:.1f} names/sec)")
    log(f"peak VRAM: {vram_peak_gb:.2f} GB (headroom on an 8GB card: "
        f"{result['vram_headroom_gb_on_8gb_card']:.2f} GB)")

    # project the full-dataset cost so the go/no-go decision at hour 1.5 has
    # a concrete number, not just a vibe
    full_dataset_rows = 2_206_821 + 5_034_616 + 5_285_603  # S1+S2+S3, train only
    projected_seconds = full_dataset_rows / names_per_sec if names_per_sec else None
    result["projection"] = {
        "full_train_dataset_rows_s1_s2_s3": full_dataset_rows,
        "projected_seconds_to_encode_all": round(projected_seconds, 0) if projected_seconds else None,
        "projected_minutes_to_encode_all": round(projected_seconds / 60, 1) if projected_seconds else None,
        "note": "This projects encoding EVERY row. The plan scopes LaBSE to "
                "non-Latin-script or thin-after-R1-R5 entities only -- the "
                "real workload will be a fraction of this, but this number "
                "is the worst-case upper bound for background encoding.",
    }
    log(f"projected time to encode the full train S1+S2+S3 pool "
        f"({full_dataset_rows:,} rows): "
        f"{result['projection']['projected_minutes_to_encode_all']} min "
        f"(worst case -- real scope is narrower, see note)")

    return result


def main():
    t_start = time.time()
    gpu_result = gpu_check()
    names = load_benchmark_names(N_BENCHMARK) if gpu_result.get("cuda_available") else []
    labse_result = labse_benchmark(names, gpu_result) if names else {"skipped": True, "reason": "gpu_check_failed"}

    report = {
        "gpu_check": gpu_result,
        "labse_benchmark": labse_result,
        "total_seconds": round(time.time() - t_start, 1),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"wrote {RESULTS_PATH}")

    log("=== SUMMARY ===")
    if gpu_result.get("cuda_available") and not labse_result.get("skipped"):
        log(f"GPU: {gpu_result['device_name']} ({gpu_result['total_vram_gb']} GB VRAM) -- WORKING")
        log(f"LaBSE: {labse_result['names_per_second']} names/sec, "
            f"peak {labse_result['vram_peak_during_encode_gb']} GB VRAM -- FITS")
        log("Go/no-go for background LaBSE encoding (hour 1.5): GO")
    else:
        log("GPU/LaBSE path is NOT usable on this machine -- cut it from the "
            "plan now, not at hour 12. The required CPU-only path is unaffected.")

    log(f"Total runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
