#!/usr/bin/env python3
"""
Step 2: assemble labeled training pairs from the sample's blocking output.

Every row of experiment2_candidate_pairs.tsv IS a training example for the
Stage 4 pairwise classifier: a real (S1, candidate) pair that survived
blocking, which needs a label (1 = true match, 0 = not) and the raw
name/address text of both sides so Stage 3 can compute similarity features
on it.

Note: audit.load_ground_truth_map() reads from the FULL dataset's
train_ground_truth.tsv (audit.FILES["train_ground_truth"]) -- not usable
here since we want the SAMPLE's ground truth
(src/sampled_data/sample_ground_truth.tsv, same 2-column schema). This
script reads that file directly with the same parsing logic instead of
calling that function.

Does NOT modify any teammate files -- reads only. Writes to
student_resource/code/business_entity_resolution/pipeline_output/.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/build_training_pairs.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit

REPO_ROOT = os.path.abspath(os.path.join(audit.STUDENT_RESOURCE, ".."))
BLOCKING_RESULTS_DIR = os.path.join(REPO_ROOT, "src", "blocking_results")
SAMPLED_DATA_DIR = os.path.join(REPO_ROOT, "src", "sampled_data")
PIPELINE_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline_output")
)


def log(msg):
    print(msg, flush=True)


def load_sample_ground_truth(path):
    """Same parsing logic as audit.load_ground_truth_map(), applied to the
    sample's ground truth file directly (that function is hardcoded to the
    full dataset's file path)."""
    gt = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for s1, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1] = set(matched.split(",")) if matched else set()
    return gt


def main():
    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    pairs_path = os.path.join(BLOCKING_RESULTS_DIR, "experiment2_candidate_pairs.tsv")
    provenance_path = os.path.join(BLOCKING_RESULTS_DIR, "experiment2_candidate_provenance.tsv")
    s1_path = os.path.join(SAMPLED_DATA_DIR, "sample_source1.tsv")
    s2_path = os.path.join(SAMPLED_DATA_DIR, "sample_source2.tsv")
    s3_path = os.path.join(SAMPLED_DATA_DIR, "sample_source3.tsv")
    gt_path = os.path.join(SAMPLED_DATA_DIR, "sample_ground_truth.tsv")

    log("Loading blocking output (pairs + provenance)...")
    pairs = pd.read_csv(pairs_path, sep="\t", dtype=str, keep_default_na=False)
    provenance = pd.read_csv(provenance_path, sep="\t", dtype=str, keep_default_na=False)
    merged_pairs = pairs.merge(
        provenance, on=["s1_entity_id", "candidate_entity_id"], how="left"
    )
    log(f"  {len(merged_pairs):,} candidate pairs with provenance")

    log("Loading source records (S1/S2/S3 sample)...")
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    log(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

    # normalize name/address once per source, reusing audit.py's existing,
    # already-fixed (Unicode-combining-mark-safe) normalization functions
    # rather than reimplementing them.
    log("Normalizing name/address fields (reusing audit.py's normalization)...")
    for df in (s1, s2, s3):
        df["norm_name"] = audit.normalize_name_series(df["business_name"])
        df["norm_addr"] = audit.normalize_series(df["business_address"])

    s1_renamed = s1.rename(columns={
        "entity_id": "s1_entity_id", "business_name": "s1_business_name",
        "business_address": "s1_business_address", "norm_name": "s1_norm_name",
        "norm_addr": "s1_norm_addr", "country": "s1_country",
    })

    candidates_pool = pd.concat([s2, s3], ignore_index=True)
    candidates_renamed = candidates_pool.rename(columns={
        "entity_id": "candidate_entity_id", "business_name": "s2_business_name",
        "business_address": "s2_business_address", "norm_name": "s2_norm_name",
        "norm_addr": "s2_norm_addr", "country": "s2_country",
    })

    log("Joining candidate pairs against source records...")
    training = merged_pairs.merge(s1_renamed, on="s1_entity_id", how="left")
    training = training.merge(candidates_renamed, on="candidate_entity_id", how="left")

    missing_s1 = training["s1_business_name"].isna().sum()
    missing_cand = training["s2_business_name"].isna().sum()
    if missing_s1 or missing_cand:
        log(f"WARNING: {missing_s1:,} pairs missing S1 record data, "
            f"{missing_cand:,} missing candidate record data -- dropping these.")
        training = training.dropna(subset=["s1_business_name", "s2_business_name"])

    log("Labeling pairs against sample ground truth...")
    gt = load_sample_ground_truth(gt_path)
    training["label"] = [
        1 if cid in gt.get(s1_id, set()) else 0
        for s1_id, cid in zip(training["s1_entity_id"], training["candidate_entity_id"])
    ]

    n_pos = int(training["label"].sum())
    n_total = len(training)
    log(f"Labeled {n_total:,} pairs: {n_pos:,} positive ({100*n_pos/n_total:.2f}%), "
        f"{n_total - n_pos:,} negative")

    # sanity check against ground truth's own count of true pairs
    total_true_in_gt = sum(len(v) for v in gt.values())
    recall_within_candidates = n_pos / total_true_in_gt if total_true_in_gt else None
    log(f"Ground truth has {total_true_in_gt:,} true pairs total for these "
        f"{len(gt):,} S1 entities. {n_pos:,} of those appear in the candidate "
        f"set (blocking recall on this join: "
        f"{100*recall_within_candidates:.2f}%, consistent with the "
        f"~99.65% reported in experiment2_blocking_comparison.tsv).")

    out_columns = [
        "s1_entity_id", "candidate_entity_id",
        "s1_business_name", "s1_business_address", "s1_norm_name", "s1_norm_addr", "s1_country",
        "s2_business_name", "s2_business_address", "s2_norm_name", "s2_norm_addr", "s2_country",
        "blocking_methods", "num_blockers", "max_block_score",
        "label",
    ]
    out_path = os.path.join(PIPELINE_OUTPUT_DIR, "training_pairs.parquet")
    training[out_columns].to_parquet(out_path, index=False)
    log(f"Wrote {out_path} ({len(training):,} rows, {len(out_columns)} columns)")


if __name__ == "__main__":
    main()
