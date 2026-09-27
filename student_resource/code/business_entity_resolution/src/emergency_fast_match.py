#!/usr/bin/env python3
"""
EMERGENCY DEADLINE FALLBACK -- built under extreme time pressure when the
full 4-method blocking pipeline proved too slow/memory-heavy to complete
on the full ~1.73M-entity test set within the remaining time before
submission close.

Strategy: pure vectorized pandas merges, no per-row Python loops, no
inverted indexes built row-by-row, no fuzzy string matching. Match on
(normalized_name, country) exact equality, and separately on
(normalized_name_prefix, country) as a looser fallback tier -- both are
single pandas .merge() calls, which run as compiled vectorized operations
across the whole DataFrame at once rather than 1.73M individual Python
function calls. This trades recall (misses near-duplicates, typos,
transliteration variants -- everything the full pipeline's TF-IDF/fuzzy/
token blockers exist to catch) for guaranteed completion time.

Produces a COMPLETE, VALID submission for every test S1 entity (matches
where found via exact/near-exact merge, empty otherwise) -- a real,
scoreable, valid-format submission beats a more sophisticated pipeline
that doesn't finish before the deadline.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/emergency_fast_match.py
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache
from pipeline_common import PIPELINE_OUTPUT_DIR, FINAL_OUTPUT_DIR, log
from finalize_candidate_pairs import finalize_candidate_pairs

OUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "emergency_fast_match")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    log("Loading normalized test_source1/2/3 (from cache, already built)...")
    s1 = dataset_cache.load_normalized(
        "test_source1", columns=["entity_id", "business_name", "business_address", "country", "norm_name"]
    )
    s2 = dataset_cache.load_normalized(
        "test_source2", columns=["entity_id", "business_name", "business_address", "country", "norm_name"]
    )
    s3 = dataset_cache.load_normalized(
        "test_source3", columns=["entity_id", "business_name", "business_address", "country", "norm_name"]
    )
    log(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    required_ids = s1["entity_id"].tolist()

    # Tier 1: EXACT normalized-name + country match -- highest confidence,
    # vectorized merge (not a per-row loop).
    log("Tier 1: exact (norm_name, country) merge...")
    t0 = time.time()
    s1_key = s1[["entity_id", "norm_name", "country"]].rename(columns={"entity_id": "s1_entity_id"})
    pool_key = pool[["entity_id", "norm_name", "country"]].rename(columns={"entity_id": "candidate_entity_id"})
    # drop empty-name rows from the join -- an empty norm_name would
    # otherwise match every other empty-name row, a false-positive flood
    s1_key_nonempty = s1_key[s1_key["norm_name"] != ""]
    pool_key_nonempty = pool_key[pool_key["norm_name"] != ""]
    exact_matches = s1_key_nonempty.merge(pool_key_nonempty, on=["norm_name", "country"], how="inner")
    exact_matches = exact_matches[["s1_entity_id", "candidate_entity_id"]]
    exact_matches["tier"] = "exact_name_country"
    log(f"  {len(exact_matches):,} exact matches in {round(time.time()-t0,1)}s")

    # Tier 2 (name-prefix8 + country) was ATTEMPTED and ABORTED: an 8-char
    # prefix is far too common across ~10M business names to be
    # discriminative -- the join blew up to a ~3.9 BILLION-row intermediate
    # result (confirmed by a real MemoryError: "Unable to allocate 29.0 GiB
    # for an array with shape (3896256673,)"), which would never finish
    # let alone fit in memory. Cut entirely under deadline pressure --
    # exact-match (Tier 1) is the only tier in this emergency fallback.
    log("Combining/deduplicating exact matches (Tier 2 prefix-match cut -- "
        "caused a 3.9B-row join blowup, not viable in remaining time)...")
    all_matches = exact_matches.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"])
    log(f"  {len(all_matches):,} unique (s1, candidate) pairs total")

    # cap per-S1 candidate count to avoid pathological blowup from a very
    # common short prefix (e.g. many companies starting with the same word)
    MAX_CANDIDATES_PER_S1 = 50
    counts_per_s1 = all_matches.groupby("s1_entity_id").size()
    oversized_s1 = counts_per_s1[counts_per_s1 > MAX_CANDIDATES_PER_S1].index
    if len(oversized_s1):
        log(f"  capping {len(oversized_s1):,} S1 entities with >{MAX_CANDIDATES_PER_S1} "
            f"candidates down to their top {MAX_CANDIDATES_PER_S1} (by tier priority: exact first)")
        # exact matches sort first (tier order), keep first N per S1 after that ordering
        all_matches["tier_rank"] = (all_matches["tier"] == "exact_name_country").astype(int)
        all_matches = all_matches.sort_values(["s1_entity_id", "tier_rank"], ascending=[True, False])
        all_matches = all_matches.groupby("s1_entity_id").head(MAX_CANDIDATES_PER_S1)

    log("Building final candidate_pairs.tsv and matching_results.tsv...")
    candidate_pairs_final = finalize_candidate_pairs(
        all_matches, required_ids, s1_col="s1_entity_id", cid_col="candidate_entity_id"
    )
    candidate_pairs_path = os.path.join(FINAL_OUTPUT_DIR, "candidate_pairs.tsv")
    candidate_pairs_final.to_csv(candidate_pairs_path, sep="\t", index=False)
    log(f"  wrote {candidate_pairs_path} ({len(candidate_pairs_final):,} rows)")

    # For matching_results.tsv under this emergency fallback: this tier
    # (exact norm_name+country match) IS the only tier, so matches = all
    # candidates. CRITICAL FIX (caught by the validator's own subset-check
    # warning on the first run of this script): must build this from
    # `all_matches` -- the SAME, capped-at-50-per-S1 dataframe
    # candidate_pairs.tsv was built from -- not from the original uncapped
    # `exact_matches`. Building it from the uncapped source meant 320,616
    # S1 entities got matched_entity_ids that were NEVER capped into their
    # own candidate_pairs.tsv row, violating the README's explicit subset
    # rule (matched ids must be a subset of that S1's candidates). The
    # validator only warns (does not reject) on this, but it's a real
    # correctness bug worth fixing rather than shipping with a known
    # violation of the stated rule.
    matching_final = finalize_candidate_pairs(
        all_matches, required_ids, s1_col="s1_entity_id", cid_col="candidate_entity_id"
    ).rename(columns={"candidate_entity_ids": "matched_entity_ids"})
    matching_results_path = os.path.join(FINAL_OUTPUT_DIR, "matching_results.tsv")
    matching_final.to_csv(matching_results_path, sep="\t", index=False)
    n_with_matches = (matching_final["matched_entity_ids"] != "").sum()
    log(f"  wrote {matching_results_path} ({len(matching_final):,} rows, "
        f"{n_with_matches:,} with >=1 match)")

    log("")
    log("=== Validating against utils/validate_submission.py ===")
    from pipeline_common import run_official_validator
    s1_ids_df = s1[["entity_id"]]
    run_official_validator(matching_results_path, candidate_pairs_path, s1_ids_df)

    log(f"\nTotal runtime: {round(time.time()-t_start,1)}s")


if __name__ == "__main__":
    main()
