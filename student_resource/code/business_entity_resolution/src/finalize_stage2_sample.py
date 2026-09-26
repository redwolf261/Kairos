#!/usr/bin/env python3
"""
Step 1 of closing the Stage 2 format gap, on the 1,000-S1 sample.

Reads the teammate's blocking output (src/blocking_results/
experiment2_candidate_pairs.tsv + experiment2_candidate_provenance.tsv --
flat, one row per (S1, candidate) pair, per schema_contracts.py's
CANDIDATE_PAIRS_INTERMEDIATE_FORMAT_SEEN_IN_REPO) and converts it into the
challenge's required candidate_pairs.tsv shape using the ALREADY-BUILT
finalize_candidate_pairs() function -- proving that tool actually closes
the gap flagged when this file was pulled, on real data, not synthetic
test rows.

Does NOT modify any of the teammate's files -- only reads them. Writes a
new file to student_resource/code/business_entity_resolution/pipeline_output/
(a new directory, not the teammate's blocking_results/).

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/finalize_stage2_sample.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
from finalize_candidate_pairs import finalize_candidate_pairs

REPO_ROOT = os.path.abspath(os.path.join(audit.STUDENT_RESOURCE, ".."))
BLOCKING_RESULTS_DIR = os.path.join(REPO_ROOT, "src", "blocking_results")
SAMPLED_DATA_DIR = os.path.join(REPO_ROOT, "src", "sampled_data")

PIPELINE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline_output")
PIPELINE_OUTPUT_DIR = os.path.normpath(PIPELINE_OUTPUT_DIR)


def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    candidate_pairs_path = os.path.join(BLOCKING_RESULTS_DIR, "experiment2_candidate_pairs.tsv")
    provenance_path = os.path.join(BLOCKING_RESULTS_DIR, "experiment2_candidate_provenance.tsv")
    s1_sample_path = os.path.join(SAMPLED_DATA_DIR, "sample_source1.tsv")

    for p in (candidate_pairs_path, provenance_path, s1_sample_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected input not found: {p}")

    log(f"Loading flat candidate pairs from {candidate_pairs_path}...")
    flat_pairs = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
    log(f"  {len(flat_pairs):,} rows, columns: {list(flat_pairs.columns)}")

    log(f"Loading provenance from {provenance_path}...")
    provenance = pd.read_csv(provenance_path, sep="\t", dtype=str, keep_default_na=False)
    log(f"  {len(provenance):,} rows, columns: {list(provenance.columns)}")

    # sanity: confirm the two files share the exact same (s1, candidate) key
    # set, per the Explore agent's earlier finding -- verify it ourselves
    # rather than trust that finding blindly.
    pairs_keys = set(zip(flat_pairs["s1_entity_id"], flat_pairs["candidate_entity_id"]))
    prov_keys = set(zip(provenance["s1_entity_id"], provenance["candidate_entity_id"]))
    if pairs_keys != prov_keys:
        only_in_pairs = pairs_keys - prov_keys
        only_in_prov = prov_keys - pairs_keys
        log(f"WARNING: candidate_pairs and provenance key sets differ! "
            f"{len(only_in_pairs)} only in pairs, {len(only_in_prov)} only in provenance. "
            f"Proceeding using candidate_pairs.tsv as the source of truth for "
            f"which pairs exist.")
    else:
        log(f"  confirmed: candidate_pairs and provenance share the same "
            f"{len(pairs_keys):,} (s1, candidate) key set")

    log(f"Loading required S1 ids from {s1_sample_path}...")
    s1_sample = pd.read_csv(s1_sample_path, sep="\t", dtype=str, keep_default_na=False)
    required_ids = s1_sample["entity_id"].tolist()
    log(f"  {len(required_ids):,} required S1 entities")

    log("Finalizing into the required candidate_pairs.tsv shape...")
    result = finalize_candidate_pairs(
        flat_pairs, required_ids,
        s1_col="s1_entity_id", cid_col="candidate_entity_id",
    )

    n_with_candidates = (result["candidate_entity_ids"] != "").sum()
    log(f"Finalized: {len(result):,} rows "
        f"({n_with_candidates:,} with >=1 candidate, "
        f"{len(result) - n_with_candidates:,} empty)")

    out_path = os.path.join(PIPELINE_OUTPUT_DIR, "candidate_pairs_sample.tsv")
    result.to_csv(out_path, sep="\t", index=False)
    log(f"Wrote {out_path}")

    # --- build-time verification: run the real challenge validator against
    # this file (as the --candidate arg) using the sample's own S1 list as
    # the "test set" -- proves the finalized output is structurally valid,
    # not just documented as such.
    log("")
    log("=== Verifying against utils/validate_submission.py ===")
    validate_against_sample(out_path, s1_sample_path)


def validate_against_sample(candidate_path, s1_sample_path):
    """Runs the challenge's own validator against the finalized file. We
    build a matching_results.tsv placeholder too (all-empty), since the
    validator requires one -- this step only checks candidate_pairs.tsv's
    OWN structural validity (required rows present, no dup ids, only
    S2-/S3- ids, no duplicate S1 rows), not match quality."""
    utils_dir = os.path.join(audit.STUDENT_RESOURCE, "utils")
    sys.path.insert(0, utils_dir)
    import validate_submission as validator

    s1_df = pd.read_csv(s1_sample_path, sep="\t", dtype=str, keep_default_na=False)
    placeholder_matching = s1_df[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
    placeholder_matching["matched_entity_ids"] = ""
    matching_path = os.path.join(PIPELINE_OUTPUT_DIR, "_placeholder_matching_for_validation.tsv")
    placeholder_matching.to_csv(matching_path, sep="\t", index=False)

    # the validator's required-S1 list comes from a directory containing
    # test_source1.tsv -- point it at a temp dir with the sample's S1 file
    # under that exact filename.
    temp_test_dir = os.path.join(PIPELINE_OUTPUT_DIR, "_temp_test_dir_for_validation")
    os.makedirs(temp_test_dir, exist_ok=True)
    temp_s1_path = os.path.join(temp_test_dir, "test_source1.tsv")
    s1_df.to_csv(temp_s1_path, sep="\t", index=False)

    errors, warnings = validator.validate(matching_path, candidate_path, temp_test_dir)

    for w in warnings:
        log(f"WARNING: {w}")
    if errors:
        log(f"FAIL -- {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
        raise SystemExit(1)

    log("PASS -- candidate_pairs_sample.tsv is structurally valid per the "
        "challenge's own validator.")

    # cleanup temp files (not part of the real output)
    os.remove(matching_path)
    os.remove(temp_s1_path)
    os.rmdir(temp_test_dir)


if __name__ == "__main__":
    main()
