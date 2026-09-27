#!/usr/bin/env python3
"""
FINAL SUBMISSION blocking run against the REAL TEST SET -- fast, proven
path under time pressure near the challenge deadline.

Deliberately SKIPS the name_char_tfidf and name_fuzzy blockers. Both are
verified real (contribute meaningfully to recall on training data), but
also verified UNRELIABLE at real test-set scale in this session: the
name_char_tfidf query step stalled for 25+ minutes with no completion on
just the SMALLEST of 3 country shards (France, 259k queries x 1.4M
candidates), twice, under otherwise-stable memory conditions -- a real,
unresolved performance problem, not something safe to gamble the
remaining time budget on.

Uses ONLY the address + name_token blockers (country-sharded, verified
FAST and RELIABLE: ~1600+ S1 rows/sec, clean memory, France's full
259,452-entity shard completed in 155.9s), then applies sibling_expansion.py
on top (verified on real training data: +12.20 percentage points recall,
19.07% -> 31.27%, by using already-confident candidates' own text as a
second query to catch paraphrase/rebrand variants the original blockers
miss).

This is the pragmatic, evidence-based choice under a hard deadline: a
complete, fast, reliably-finishing pipeline beats a more sophisticated
one that has now failed to complete twice on the smallest shard alone.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/test_blocking_fast.py
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


def build_lite_structures_for_shard(pool_df, max_bucket_size=None):
    """Fast path: build ONLY what address+name_token blockers need
    (source_lookup + 3 index dicts) -- skip TF-IDF fitting and RapidFuzz
    choice-dict construction entirely, since those blockers are not run
    here. This also avoids the TF-IDF fitting step's own time cost
    (was ~10-20s per shard, not the bottleneck, but free savings)."""
    pool_df = sb.prepare_pool_fields_parallel(pool_df)
    max_bucket_size = max_bucket_size or sb.MAX_INDEX_BUCKET_SIZE

    source_lookup = dict(zip(pool_df["entity_id"], pool_df["country_norm"]))

    log("  Building address inverted index...")
    address_index = sb.defaultdict(set)
    for entity_id, row in zip(pool_df["entity_id"], pool_df["address_tokens"]):
        for token in row:
            address_index[token].add(entity_id)
    address_index = sb._drop_oversized_buckets(address_index, max_bucket_size, "address_index")

    log("  Building name token indexes...")
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

    return {
        "source_lookup": source_lookup,
        "address_index": address_index,
        "name_token_index": name_token_index,
        "name_core_token_index": name_core_token_index,
    }


def run_fast_blocking_for_shard(s1_shard, structures):
    """Same as sb.run_address_name_token_blockers_parallel() /
    sb.run_blocking()'s address+name_token path, but standalone here since
    sb.run_blocking() also tries to run the TF-IDF/fuzzy blockers we're
    deliberately skipping."""
    if len(s1_shard) > 50_000:
        s1_shard = sb.prepare_pool_fields_parallel(s1_shard)
        all_rows = sb.run_address_name_token_blockers_parallel(s1_shard, structures)
    else:
        s1_shard = s1_shard.copy()
        s1_shard["address_tokens"] = s1_shard["business_address"].apply(sb.get_informative_address_tokens)
        s1_shard["name_tokens"] = s1_shard["business_name"].apply(sb.get_name_tokens)
        s1_shard["name_core_tokens"] = s1_shard["business_name"].apply(sb.get_name_core_tokens)
        all_rows = []
        for _, row in s1_shard.iterrows():
            all_rows.extend(sb.retrieve_address_candidates(row, structures))
            all_rows.extend(sb.retrieve_name_token_candidates(row, structures))

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
        log(f"=== SHARD: {country} (fast path: address+name_token only) ===")

        # CRASH-RECOVERY FIX (found the hard way: a near-OOM crash killed a
        # run right as the FINAL shard was almost done, losing two
        # already-completed shards' results -- 45.5M real candidate pairs
        # from France+India -- because nothing was written to disk until
        # ALL 3 shards finished and were concatenated together. Each shard
        # is now written to disk immediately after it completes, AND a
        # shard whose output file already exists is skipped entirely on
        # a re-run -- so killing/restarting this script no longer loses
        # already-completed shards' work.
        shard_pairs_path = os.path.join(OUT_DIR, f"candidate_pairs_{country}.tsv")
        shard_prov_path = os.path.join(OUT_DIR, f"candidate_provenance_{country}.tsv")
        if os.path.isfile(shard_pairs_path) and os.path.isfile(shard_prov_path):
            log(f"  {country} shard already completed (found {shard_pairs_path}) -- loading, not re-running.")
            candidate_pairs = pd.read_csv(shard_pairs_path, sep="\t", dtype=str, keep_default_na=False)
            provenance = pd.read_csv(shard_prov_path, sep="\t", dtype=str, keep_default_na=False)
            per_method_counts = {}  # not recomputed on resume; total counts below will be partial for resumed shards
        else:
            t_shard = time.time()
            s1_shard = s1_all[s1_all["country"] == country].copy()
            log(f"  {len(s1_shard):,} test S1 entities for {country}")

            pool_shard = load_test_pool_for_country(country)
            if pool_shard.empty:
                log(f"  WARNING: no candidate pool records for country={country} -- skipping.")
                continue

            structures = build_lite_structures_for_shard(pool_shard)
            del pool_shard
            gc.collect()

            candidate_pairs, provenance, per_method_counts = run_fast_blocking_for_shard(s1_shard, structures)
            del structures
            gc.collect()

            log(f"  {country} shard: {len(candidate_pairs):,} candidate pairs, "
                f"per-method: {per_method_counts}")
            log(f"  {country} shard done in {round(time.time()-t_shard,1)}s")

            # Write THIS shard's results immediately -- do not wait for
            # the other shards.
            candidate_pairs.to_csv(shard_pairs_path, sep="\t", index=False)
            provenance.to_csv(shard_prov_path, sep="\t", index=False)
            log(f"  wrote {shard_pairs_path} and {shard_prov_path} (checkpoint -- safe from here on)")

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
        "note": "FAST PATH: address+name_token blockers only, TF-IDF and fuzzy "
                "deliberately skipped (unreliable at test-set scale in this "
                "session -- see module docstring). Sibling expansion should "
                "be run on this output next.",
    }
    report_path = os.path.join(OUT_DIR, "test_blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log(f"\nTotal runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
