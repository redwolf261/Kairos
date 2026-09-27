#!/usr/bin/env python3
"""
Country-sharded full test-set blocking -- implements the hardware-aware
plan doc's Stage 2 "master rule" directly:

    "partition by normalized country first, and process one country shard
    completely -- build, run, extract results, discard the matrix --
    before starting the next. Never hold two countries' matrices in
    memory simultaneously."

This is the fix for the actual root cause behind every crash tonight:
test_blocking.py (the non-sharded version) builds indexes/TF-IDF/RapidFuzz
structures for ALL 3 countries (France, India, US) simultaneously, then
runs blocking against all of them. At the real test-set pool size
(~10M rows across 3 countries), holding all 3 countries' structures in
memory at once is what pushed free RAM to critical levels 4 times in a
row -- confirmed by direct measurement (main process 13-17GB resident on
multiple attempts). This script never builds a second country's
structures until the first country's are extracted and discarded.

Only S1 entities matching each country are blocked against that
country's shard -- correct by construction, since a US business can only
ever match a US record.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/test_blocking_sharded.py

Output: pipeline_output/test_blocking/candidate_pairs_flat.tsv (+ provenance),
same shape as test_blocking.py's output, so downstream stages don't need
to know which blocking script produced it.
"""

import gc
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache
import scaled_blocking as sb
from pipeline_common import PIPELINE_OUTPUT_DIR, log

OUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "test_blocking")


def load_test_pool_for_country(country):
    """Load ONLY this country's test_source2+3 records -- never the full
    multi-country pool. This is the core of the sharding fix: the pool
    DataFrame that exists in memory at any moment is at most one
    country's worth of records (largest shard here: India, ~7M records),
    never all 3 countries' ~10M combined."""
    log(f"  Loading test_source2+3 records for country={country}...")
    t0 = time.time()
    frames = []
    for src in ("test_source2", "test_source3"):
        for chunk in dataset_cache.load_normalized_lazy(
            src, columns=["entity_id", "business_name", "business_address", "country"]
        ):
            filtered = chunk[chunk["country"] == country]
            if len(filtered):
                frames.append(filtered.copy())
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "country"])
    log(f"    {len(pool):,} records loaded in {round(time.time()-t0,1)}s")
    return pool


def build_structures_for_shard(pool_df):
    """Same structure-building logic as scaled_blocking.build_structures(),
    but pool_df here is ALREADY single-country -- so every dict/index/
    TF-IDF-vectorizer built is sized to one shard, never the full pool.
    This reuses build_structures() as-is (it already loops
    `for country, group in pool_df.groupby(...)`, which is a no-op loop
    of exactly 1 iteration when pool_df is single-country -- no code
    duplication needed for the per-country logic itself)."""
    return sb.build_structures(pool_df)


def run_blocking_for_shard(s1_shard, structures):
    """Same driver as sb.run_blocking(), reused as-is -- s1_shard here is
    already filtered to the one country these structures cover."""
    return sb.run_blocking(s1_shard, structures)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    log("Loading ALL test_source1.tsv entities (every one needs a submission row)...")
    s1_all = dataset_cache.load_normalized(
        "test_source1", columns=["entity_id", "business_name", "business_address", "country"]
    )
    log(f"  {len(s1_all):,} test S1 entities")

    countries = sorted(s1_all["country"].unique())
    log(f"Countries present: {countries}")

    all_pairs_frames = []
    all_provenance_frames = []
    per_method_counts_total = {}

    for country in countries:
        log("")
        log(f"=== SHARD: {country} ===")
        t_shard = time.time()

        s1_shard = s1_all[s1_all["country"] == country].copy()
        log(f"  {len(s1_shard):,} test S1 entities for {country}")

        pool_shard = load_test_pool_for_country(country)
        if pool_shard.empty:
            log(f"  WARNING: no candidate pool records for country={country} -- "
                f"all {len(s1_shard):,} S1 entities in this shard will have 0 candidates.")
            # still need to emit rows for these S1 ids (all empty) --
            # handled by finalize_candidate_pairs() downstream via
            # required_ids covering every test S1 id regardless of
            # whether it appears in all_pairs_frames.
            continue

        structures = build_structures_for_shard(pool_shard)
        del pool_shard
        gc.collect()

        candidate_pairs, provenance, per_method_counts = run_blocking_for_shard(s1_shard, structures)

        # CRITICAL: discard this shard's structures before the next
        # country starts -- this IS the fix. Explicit del + gc.collect()
        # rather than relying on scope exit, since these dicts/matrices
        # can be large enough that prompt collection matters.
        del structures
        gc.collect()

        log(f"  {country} shard: {len(candidate_pairs):,} candidate pairs, "
            f"per-method: {per_method_counts}")
        log(f"  {country} shard done in {round(time.time()-t_shard,1)}s")

        all_pairs_frames.append(candidate_pairs)
        all_provenance_frames.append(provenance)
        for method, count in per_method_counts.items():
            per_method_counts_total[method] = per_method_counts_total.get(method, 0) + count

    log("")
    log("=== Combining all shards ===")
    candidate_pairs_final = pd.concat(all_pairs_frames, ignore_index=True) if all_pairs_frames else pd.DataFrame(
        columns=["s1_entity_id", "candidate_entity_id"])
    provenance_final = pd.concat(all_provenance_frames, ignore_index=True) if all_provenance_frames else pd.DataFrame(
        columns=["s1_entity_id", "candidate_entity_id", "blocking_methods", "num_blockers", "max_block_score"])

    log(f"Total candidate pairs across all shards: {len(candidate_pairs_final):,} "
        f"(avg {len(candidate_pairs_final)/len(s1_all):.1f} candidates/S1)")
    log(f"Per-method raw row counts (summed across shards): {per_method_counts_total}")

    candidate_pairs_path = os.path.join(OUT_DIR, "candidate_pairs_flat.tsv")
    provenance_path = os.path.join(OUT_DIR, "candidate_provenance.tsv")
    candidate_pairs_final.to_csv(candidate_pairs_path, sep="\t", index=False)
    provenance_final.to_csv(provenance_path, sep="\t", index=False)
    log(f"Wrote {candidate_pairs_path}")
    log(f"Wrote {provenance_path}")

    report = {
        "n_s1_total": len(s1_all),
        "countries": countries,
        "n_candidate_pairs": len(candidate_pairs_final),
        "avg_candidates_per_s1": round(len(candidate_pairs_final) / len(s1_all), 2),
        "per_method_raw_row_counts": per_method_counts_total,
        "total_seconds": round(time.time() - t_start, 1),
        "sharding": "country-sharded (one country's structures in memory at a time, "
                    "per the hardware-aware plan doc's Stage 2 master rule)",
    }
    report_path = os.path.join(OUT_DIR, "test_blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log(f"\nTotal runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
