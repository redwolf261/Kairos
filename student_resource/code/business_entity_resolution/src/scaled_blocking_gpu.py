#!/usr/bin/env python3
"""
GPU variant of scaled_blocking.py -- SAME overall blocking design (union
of several independent blockers by (s1_id, candidate_id), same provenance
schema), same sampling/pool-loading/recall-measurement code (imported
directly from scaled_blocking.py, not copy-pasted, so there is exactly one
source of truth for everything that doesn't change here).

WHAT CHANGES: the two RAM-heavy blockers (name_char_tfidf: per-country
TfidfVectorizer + sp_matmul_topn; name_fuzzy: per-country RapidFuzz
process.cdist) are REPLACED by a single name_embedding blocker: LaBSE
sentence embeddings (already verified working on this machine's GPU --
see gpu_check_labse_benchmark.py, 1966.5 names/sec, 2.18GB peak VRAM on an
8.52GB card) + GPU top-K cosine similarity search. Same role in the
pipeline (find plausible name matches beyond exact token overlap), same
per-country sharding, same top-K/threshold gating pattern as the blockers
it replaces -- just computed on the GPU instead of as CPU sparse-matrix/
string-edit-distance math.

WHY: three separate CPU/RAM memory bugs were found and fixed in
scaled_blocking.py tonight (pool-prep worker sizing, TF-IDF vocabulary
size, TF-IDF query-side chunking), and even after all three fixes the
machine's available system RAM behaved inconsistently across repeated
full-scale runs (clean with ~19GB free some runs, dangerously tight at
~2.8GB free on others, same code/scale) -- likely Windows-level memory
pressure/fragmentation building up across many large CPU-side allocations
in one long session. LaBSE embeddings are small, FIXED-SIZE, dense vectors
(768 floats/name) with no vocabulary-size or string-length blowup risk at
all, and the encode step runs on the GPU's own 8.52GB VRAM (5.82GB
headroom measured), moving the heaviest computation off system RAM
entirely instead of trying to further bound it there.

MEMORY DESIGN (revised after TWO real near-misses caught while testing,
not just worked out on paper -- see git history for the first version's
docstring, which assumed encoding all countries up front was fine):
  - Pool embeddings do NOT fit in VRAM at full scale: 10.32M rows x 768
    dims x 4 bytes (float32) = ~31.7GB, and even per-country (US ~6.19M
    rows = ~19GB, India ~4.13M rows = ~12.7GB) that's still bigger than
    the 8.52GB card. So pool embeddings are computed on GPU (fast) but
    STORED in system RAM as float16 (halves the float32 size: US ~9.5GB,
    India ~6.3GB).
  - CRITICAL, verified by an actual near-miss: US ~9.5GB + India ~6.3GB
    held SIMULTANEOUSLY (~15.85GB combined) does NOT reliably fit
    alongside everything else already resident on this machine (indexes,
    the LaBSE model, Python/CUDA runtime overhead) -- measured to push the
    process to ~14.6GB resident with under 800MB system free while US was
    still mid-encode (India's embeddings already sitting in RAM the whole
    time). Fixed by NEVER encoding more than one country's pool at a time:
    build_structures_gpu() only stores each country's raw (unencoded) pool
    text; retrieve_name_embedding_candidates_batch() encodes one country's
    pool, uses it for that country's queries, then frees it (RAM array AND
    GPU tensor) before moving to the next country -- exactly the
    "partition by country, process one shard completely before starting
    the next" principle the hardware-aware plan doc already recommended
    for TF-IDF, which the first version of this file failed to apply here.
  - Query-time similarity search streams query embedding batches TO the
    GPU against that one country's pool embeddings (moved to GPU once per
    country, not once per query chunk), computes cosine similarity + top-K
    via a matmul + torch.topk, and only the top-K (id, score) results come
    back to the CPU side -- never a dense n_queries x n_candidates result
    matrix, same principle as sp_matmul_topn's sparse output but via GPU
    top-k instead.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/scaled_blocking_gpu.py

Env vars (same names/semantics as scaled_blocking.py where applicable):
    BLOCKING_N_S1                  how many S1 entities to sample (default 100000)
    BLOCKING_SEED                  sampling seed (default 42)
    BLOCKING_LOG_EVERY             progress print interval in S1 rows (default 1000)
    BLOCKING_POOL_ROW_CAP          testing-only pool downsample (see scaled_blocking.py)
    GPU_BLOCKING_MODEL_NAME        sentence-transformers model id (default LaBSE)
    GPU_BLOCKING_ENCODE_BATCH      encoding batch size (default 256, matches the
                                   verified benchmark)
    GPU_BLOCKING_QUERY_CHUNK       how many S1 queries to send to the GPU per
                                   similarity-search call (default 2000 -- generous
                                   headroom vs. TF-IDF's 1000, since embedding
                                   query vectors are much smaller/denser than
                                   TF-IDF sparse rows)
    GPU_BLOCKING_TOP_K             candidates kept per S1 entity from this blocker
                                   (default 50, matching NAME_CHAR_TOP_K/
                                   NAME_FUZZY_TOP_K in scaled_blocking.py, since
                                   this blocker replaces both)
    GPU_BLOCKING_MIN_SIMILARITY    cosine similarity floor (default 0.5 -- a
                                   starting point between the two thresholds it
                                   replaces (0.35 char-tfidf, 0.55 fuzzy); ROUND 1
                                   should re-tune this against measured recall,
                                   not assume it transfers directly)

Output: same 4 files, same schema, as scaled_blocking.py, under
pipeline_output/scaled_blocking_gpu/ (separate directory so a run of this
script never overwrites a scaled_blocking.py run's output).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import scaled_blocking as sb  # noqa: E402 -- reuse everything that doesn't change
from pipeline_common import PIPELINE_OUTPUT_DIR, log

N_S1 = int(os.environ.get("BLOCKING_N_S1", "100000"))
SEED = int(os.environ.get("BLOCKING_SEED", "42"))
LOG_EVERY = int(os.environ.get("BLOCKING_LOG_EVERY", "1000"))

OUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "scaled_blocking_gpu")

MODEL_NAME = os.environ.get("GPU_BLOCKING_MODEL_NAME", "sentence-transformers/LaBSE")
ENCODE_BATCH = int(os.environ.get("GPU_BLOCKING_ENCODE_BATCH", "256"))
QUERY_CHUNK = int(os.environ.get("GPU_BLOCKING_QUERY_CHUNK", "2000"))
TOP_K = int(os.environ.get("GPU_BLOCKING_TOP_K", "50"))
MIN_SIMILARITY = float(os.environ.get("GPU_BLOCKING_MIN_SIMILARITY", "0.5"))


def _require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- this script requires the GPU path verified by "
            "gpu_check_labse_benchmark.py. Use scaled_blocking.py (CPU) instead."
        )
    return torch


def encode_names_gpu(model, texts, batch_size=None):
    """Encode a list of strings to LaBSE embeddings on GPU, return as a
    float16 numpy array (halves memory vs. float32 -- see module docstring
    for why this matters at full pool scale). Empty strings still get a
    real (zero-ish) embedding rather than being dropped, so row alignment
    with entity_ids/texts is never in question downstream."""
    batch_size = batch_size or ENCODE_BATCH
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embeddings.astype(np.float16)


def build_structures_gpu(pool_df, max_bucket_size=None):
    """Same as scaled_blocking.build_structures(), except the TF-IDF
    fitting step is replaced with LaBSE encoding of every pool name, once
    per country shard (same sharding pattern, same reason: bound memory
    per shard rather than one global structure)."""
    torch = _require_gpu()
    from sentence_transformers import SentenceTransformer

    pool_df = sb.prepare_pool_fields_parallel(pool_df)

    source_lookup = dict(zip(pool_df["entity_id"], pool_df["country_norm"]))
    max_bucket_size = max_bucket_size or sb.MAX_INDEX_BUCKET_SIZE

    log("Building address inverted index...")
    address_index = sb.defaultdict(set)
    for entity_id, row in zip(pool_df["entity_id"], pool_df["address_tokens"]):
        for token in row:
            address_index[token].add(entity_id)
    address_index = sb._drop_oversized_buckets(address_index, max_bucket_size, "address_index")

    log("Building name token indexes...")
    name_token_index = sb.defaultdict(set)
    name_core_token_index = sb.defaultdict(set)
    for entity_id, tokens, core_tokens in zip(
        pool_df["entity_id"], pool_df["name_tokens"], pool_df["name_core_tokens"]
    ):
        for token in set(tokens):
            if len(token) >= sb.NAME_TOKEN_MIN_LENGTH:
                name_token_index[token].add(entity_id)
        for token in set(core_tokens):
            if len(token) >= sb.NAME_TOKEN_MIN_LENGTH:
                name_core_token_index[token].add(entity_id)
    name_token_index = sb._drop_oversized_buckets(name_token_index, max_bucket_size, "name_token_index")
    name_core_token_index = sb._drop_oversized_buckets(name_core_token_index, max_bucket_size, "name_core_token_index")

    # MEMORY FIX #1 (post-OOM-near-miss on the very first full-scale GPU
    # test): pool_df was held alive with ALL its derived columns, including
    # the now-unneeded address_tokens/name_tokens/name_core_tokens token
    # LISTS (heavier than plain strings). Only entity_id/name_char_text/
    # country_norm are needed downstream (for encoding), so slim it down
    # now and free the rest.
    pool_df = pool_df[["entity_id", "name_char_text", "country_norm"]].copy()

    log(f"Loading LaBSE model '{MODEL_NAME}' on GPU...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    log(f"  model loaded in {time.time()-t0:.1f}s")

    # MEMORY FIX #2 (post-second-near-miss): the original version encoded
    # BOTH countries' pool names here and stashed both embedding arrays in
    # `embeddings_by_country`, held for the rest of the run -- India's
    # 6.35GB + US's ~9.5GB simultaneously (~15.85GB combined) on top of the
    # indexes above and everything else resident. Verified: this pushed
    # the process to ~14.6GB resident with under 800MB system free WHILE
    # US was still encoding (India's embeddings already computed and
    # sitting in RAM the whole time). Fixed by NOT encoding here at all --
    # only the slimmed pool_df (grouped by country) is kept in `structures`
    # so encoding can happen per-country, immediately followed by that
    # country's query pass, in retrieve_name_embedding_candidates_batch()
    # below, freeing each country's pool embeddings before the next
    # country's are computed. Same principle as the module docstring's
    # per-country-shard TF-IDF/hardware-aware-plan-doc guidance: process
    # one country shard completely before starting the next, never hold
    # two at once.
    pool_by_country = {
        country: group[["entity_id", "name_char_text"]].reset_index(drop=True)
        for country, group in pool_df.groupby("country_norm") if country
    }
    del pool_df

    return {
        "source_lookup": source_lookup,
        "address_index": address_index,
        "name_token_index": name_token_index,
        "name_core_token_index": name_core_token_index,
        "pool_by_country": pool_by_country,
        "labse_model": model,
    }


def retrieve_name_embedding_candidates_batch(s1_batch_df, structures):
    """GPU replacement for name_char_tfidf + name_fuzzy. Encodes the S1
    query batch on GPU, then for each country ENCODES THAT COUNTRY'S POOL
    NAMES on demand (not pre-computed for all countries up front -- see
    build_structures_gpu()'s MEMORY FIX #2), moves the resulting
    embeddings to GPU ONCE per country (not once per query chunk), streams
    query chunks against them via matmul + topk, then frees that country's
    embeddings (both the RAM-side numpy array and the GPU-side tensor)
    before moving to the next country. Peak memory is bounded by (one
    country's embeddings, in RAM AND briefly on GPU) + (one query chunk),
    never by two countries' embeddings simultaneously or by total S1
    count/pool size at once.

    Returns result dicts in the same shape as the two blockers it
    replaces, tagged with block_method="name_embedding" so provenance
    output clearly shows which method found each pair (not silently
    relabeled as one of the old names)."""
    torch = _require_gpu()
    model = structures["labse_model"]
    results = []

    for country, group in s1_batch_df.groupby("country_norm"):
        pool_for_country = structures["pool_by_country"].get(country)
        if pool_for_country is None:
            continue

        queries_df = group[group["name_char_text"] != ""]
        if queries_df.empty:
            continue
        query_ids = queries_df["entity_id"].tolist()
        query_texts = queries_df["name_char_text"].tolist()
        query_emb = encode_names_gpu(model, query_texts)  # (n_queries, 768) float16

        # Encode THIS country's pool now (on demand), use it for every
        # query chunk below, then free it (RAM numpy array AND GPU tensor)
        # before the loop moves to the next country -- never two
        # countries' pool embeddings alive at once.
        t_enc = time.time()
        pool_texts = pool_for_country["name_char_text"].fillna("").astype(str).tolist()
        pool_entity_ids = pool_for_country["entity_id"].to_numpy()
        pool_emb = encode_names_gpu(model, pool_texts)
        del pool_texts
        rate = len(pool_entity_ids) / (time.time() - t_enc) if time.time() - t_enc else 0
        log(f"    name_embedding/{country}: encoded {len(pool_entity_ids):,} pool names "
            f"in {time.time()-t_enc:.1f}s ({rate:.1f}/s), "
            f"{pool_emb.nbytes/1e9:.2f} GB in system RAM")

        # Move this country's pool embeddings to GPU once, reused across
        # every query chunk below -- NOT re-uploaded per chunk.
        pool_gpu = torch.from_numpy(pool_emb).to("cuda", dtype=torch.float16)
        del pool_emb

        n_queries = len(query_texts)
        n_chunks = -(-n_queries // QUERY_CHUNK)
        log(f"    name_embedding/{country}: {n_queries:,} queries x {len(pool_entity_ids):,} "
            f"candidates, {n_chunks} chunk(s) of <={QUERY_CHUNK}")

        for chunk_i in range(n_chunks):
            start = chunk_i * QUERY_CHUNK
            end = min(start + QUERY_CHUNK, n_queries)
            chunk_ids = query_ids[start:end]
            chunk_gpu = torch.from_numpy(query_emb[start:end]).to("cuda", dtype=torch.float16)

            # Embeddings are normalize_embeddings=True (unit-norm), so a
            # plain matmul IS cosine similarity -- no extra division needed.
            sim = chunk_gpu @ pool_gpu.T  # (chunk_size, n_pool_candidates)
            k = min(TOP_K, sim.shape[1])
            top_scores, top_idx = torch.topk(sim, k=k, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_idx = top_idx.cpu().numpy()

            for row_idx, s1_id in enumerate(chunk_ids):
                for score, col_idx in zip(top_scores[row_idx], top_idx[row_idx]):
                    if score < MIN_SIMILARITY:
                        continue
                    results.append({
                        "s1_entity_id": s1_id,
                        "candidate_entity_id": pool_entity_ids[col_idx],
                        "block_method": "name_embedding",
                        "block_score": float(score),
                    })
            del sim, top_scores, top_idx, chunk_gpu

        del pool_gpu
        torch.cuda.empty_cache()
    return results


def run_blocking_gpu(s1_sample, structures):
    """Same driver as scaled_blocking.run_blocking(), except the
    name_char_tfidf + name_fuzzy steps are replaced by the single
    name_embedding step above."""
    s1_sample = s1_sample.copy()
    s1_sample["address_tokens"] = s1_sample["business_address"].apply(sb.get_informative_address_tokens)
    s1_sample["name_tokens"] = s1_sample["business_name"].apply(sb.get_name_tokens)
    s1_sample["name_core_tokens"] = s1_sample["business_name"].apply(sb.get_name_core_tokens)
    s1_sample["name_char_text"] = s1_sample["business_name"].apply(sb.get_name_char_text)
    s1_sample["country_norm"] = s1_sample["country"].apply(lambda x: sb.normalize_text(x, transliterate=True))

    all_rows = []
    total = len(s1_sample)
    t0 = time.time()

    log("Running address + name_token blockers (per-row dict lookups)...")
    for i, (_, row) in enumerate(s1_sample.iterrows(), start=1):
        if i == 1 or i % LOG_EVERY == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate if rate else 0
            log(f"  Processing S1 {i:,}/{total:,} ({rate:.1f}/s, ETA {eta/60:.1f} min)")
        all_rows.extend(sb.retrieve_address_candidates(row, structures))
        all_rows.extend(sb.retrieve_name_token_candidates(row, structures))
    log(f"  done in {round(time.time()-t0,1)}s")

    log("Running name_embedding blocker (GPU, batched per country)...")
    t1 = time.time()
    all_rows.extend(retrieve_name_embedding_candidates_batch(s1_sample, structures))
    log(f"  done in {round(time.time()-t1,1)}s")

    log(f"Blocking done in {round(time.time()-t0,1)}s total ({total} S1 rows)")

    all_df = pd.DataFrame(all_rows, columns=["s1_entity_id", "candidate_entity_id", "block_method", "block_score"])
    candidate_pairs = all_df[["s1_entity_id", "candidate_entity_id"]].drop_duplicates().reset_index(drop=True)
    provenance = (
        all_df.groupby(["s1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(
            blocking_methods=("block_method", lambda v: ",".join(sorted(set(v)))),
            num_blockers=("block_method", "nunique"),
            max_block_score=("block_score", "max"),
        )
    )
    per_method_counts = all_df["block_method"].value_counts().to_dict()
    return candidate_pairs, provenance, per_method_counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    s1_sample = sb.sample_s1(N_S1, SEED)
    countries_needed = set(s1_sample["country"].unique())

    pool_df = sb.load_candidate_pool(countries_needed)
    candidate_pool_size = len(pool_df)
    structures = build_structures_gpu(pool_df)
    del pool_df

    candidate_pairs, provenance, per_method_counts = run_blocking_gpu(s1_sample, structures)

    log(f"Union candidate pairs: {len(candidate_pairs):,} "
        f"(avg {len(candidate_pairs)/len(s1_sample):.1f} candidates/S1)")
    log(f"Per-method raw row counts: {per_method_counts}")

    recall, total_true, hits = sb.measure_recall(candidate_pairs, s1_sample)

    candidate_pairs_path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    provenance_path = os.path.join(OUT_DIR, "candidate_provenance.tsv")
    s1_sample_path = os.path.join(OUT_DIR, "s1_sample.tsv")

    candidate_pairs.to_csv(candidate_pairs_path, sep="\t", index=False)
    provenance.to_csv(provenance_path, sep="\t", index=False)
    s1_sample[["entity_id", "business_name", "business_address", "country"]].to_csv(
        s1_sample_path, sep="\t", index=False
    )
    log(f"Wrote {candidate_pairs_path}")
    log(f"Wrote {provenance_path}")
    log(f"Wrote {s1_sample_path}")

    report = {
        "n_s1_sampled": len(s1_sample),
        "seed": SEED,
        "countries": sorted(countries_needed),
        "candidate_pool_size": candidate_pool_size,
        "n_candidate_pairs": len(candidate_pairs),
        "avg_candidates_per_s1": round(len(candidate_pairs) / len(s1_sample), 2),
        "per_method_raw_row_counts": per_method_counts,
        "total_true_pairs": total_true,
        "true_pairs_recovered": hits,
        "recall_pct": round(100 * recall, 3) if recall is not None else None,
        "total_seconds": round(time.time() - t_start, 1),
        "variant": "gpu",
        "model_name": MODEL_NAME,
        "min_similarity": MIN_SIMILARITY,
        "top_k": TOP_K,
        "deviation_from_scaled_blocking": "name_char_tfidf + name_fuzzy replaced by a "
                                           "single LaBSE-embedding name_embedding blocker "
                                           "(see module docstring)",
    }
    report_path = os.path.join(OUT_DIR, "blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log(f"\nTotal runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
