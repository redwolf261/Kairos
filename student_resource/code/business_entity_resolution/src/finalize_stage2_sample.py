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
from finalize_candidate_pairs import finalize_candidate_pairs
from pipeline_common import (
    PIPELINE_OUTPUT_DIR, BLOCKING_CANDIDATE_PAIRS_PATH, BLOCKING_PROVENANCE_PATH,
    SAMPLE_SOURCE1_PATH, log, run_official_validator,
)


def main():
    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    candidate_pairs_path = BLOCKING_CANDIDATE_PAIRS_PATH
    provenance_path = BLOCKING_PROVENANCE_PATH
    s1_sample_path = SAMPLE_SOURCE1_PATH

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
    # not just documented as such. We build an all-empty matching_results.tsv
    # placeholder too, since the validator requires one -- this step only
    # checks candidate_pairs_sample.tsv's OWN structural validity (required
    # rows present, no dup ids, only S2-/S3- ids), not match quality.
    log("")
    log("=== Verifying against utils/validate_submission.py ===")
    placeholder_matching = s1_sample[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
    placeholder_matching["matched_entity_ids"] = ""
    placeholder_matching_path = os.path.join(PIPELINE_OUTPUT_DIR, "_placeholder_matching_for_validation.tsv")
    placeholder_matching.to_csv(placeholder_matching_path, sep="\t", index=False)
    try:
        run_official_validator(placeholder_matching_path, out_path, s1_sample)
    finally:
        os.remove(placeholder_matching_path)


if __name__ == "__main__":
    main()
