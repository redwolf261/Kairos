#!/usr/bin/env python3
"""
Step 3 (Stage 3): feature engineering on the labeled training pairs.

Reuses experiments.py's ALREADY-BUILT, ALREADY-PARALLELIZED
compute_features_parallel() for the core similarity features (char3/char4
Jaccard, token Jaccard, rapidfuzz edit-similarity, token-set-ratio,
length-ratio, exact-equal flags -- for both name and address) rather than
reimplementing any of that. See schema_contracts.py's STAGE3_FEATURE_FILE_DRAFT
for the originally proposed shape; this script implements a pragmatic subset
of it -- the well-defined, cheap-to-compute features -- not the full
40-60-feature wishlist.

Adds on top of the reused similarity features:
  - route_count       = num_blockers (from provenance, already in the input)
  - max_block_score   (already in the input, passed through)
  - country_conflict  = s1_country != s2_country
  - name_missing_s1 / name_missing_candidate
  - address_missing_s1 / address_missing_candidate

Input: pipeline_output/training_pairs.parquet (from build_training_pairs.py)
Output: pipeline_output/stage3_features.parquet

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/stage3_features.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import compute_features_parallel
from pipeline_common import PIPELINE_OUTPUT_DIR, NON_FEATURE_COLS, log


def add_cheap_extra_features(df):
    """Features that don't need the parallelized similarity computation --
    cheap vectorized pandas ops, computed directly here."""
    df["route_count"] = df["num_blockers"].astype(int)
    df["max_block_score"] = df["max_block_score"].astype(float)
    df["country_conflict"] = (df["s1_country"] != df["s2_country"]).astype(int)
    df["name_missing_s1"] = (df["s1_business_name"].fillna("") == "").astype(int)
    df["name_missing_candidate"] = (df["s2_business_name"].fillna("") == "").astype(int)
    df["address_missing_s1"] = (df["s1_business_address"].fillna("") == "").astype(int)
    df["address_missing_candidate"] = (df["s2_business_address"].fillna("") == "").astype(int)
    return df


def main():
    in_path = os.path.join(PIPELINE_OUTPUT_DIR, "training_pairs.parquet")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(
            f"{in_path} not found -- run build_training_pairs.py first."
        )

    log(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)
    log(f"  {len(df):,} rows, columns: {list(df.columns)}")

    log("Computing similarity features (reusing experiments.py's "
        "compute_features_parallel)...")
    df_with_sim = compute_features_parallel(df)

    log("Adding provenance/missingness/conflict features...")
    df_final = add_cheap_extra_features(df_with_sim)

    # sanity: no NaN in the join-key or label columns; similarity features
    # should also be fully populated since compute_pair_features always
    # returns a value (empty-string inputs are handled inside
    # pair_similarity_features, not left as NaN)
    required_no_nan = ["s1_entity_id", "candidate_entity_id", "label"]
    for col in required_no_nan:
        n_nan = df_final[col].isna().sum()
        if n_nan:
            raise ValueError(f"{n_nan} NaN values found in required column '{col}'")

    feature_cols = [c for c in df_final.columns if c not in NON_FEATURE_COLS]
    n_nan_features = df_final[feature_cols].isna().sum()
    if n_nan_features.sum():
        log(f"WARNING: NaN found in feature columns: "
            f"{n_nan_features[n_nan_features > 0].to_dict()}")

    log(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # spot-check: known true matches should skew toward high similarity,
    # known non-matches toward low -- print aggregate means as a sanity
    # signal (not a hard assertion, real data has noisy matches too, per
    # Experiment 1's finding that only ~43% of true matches are exact-normalized)
    log("")
    log("=== Spot-check: mean feature values by label ===")
    check_cols = ["name_char3_jaccard", "name_edit_similarity", "name_token_set_ratio",
                  "addr_char3_jaccard", "addr_edit_similarity"]
    check_cols = [c for c in check_cols if c in df_final.columns]
    means = df_final.groupby("label")[check_cols].mean()
    log(means.to_string())
    if len(means) == 2:
        for col in check_cols:
            if means.loc[1, col] <= means.loc[0, col]:
                log(f"WARNING: '{col}' does NOT separate as expected "
                    f"(label=1 mean {means.loc[1,col]:.4f} <= label=0 mean "
                    f"{means.loc[0,col]:.4f}) -- investigate before training.")

    out_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage3_features.parquet")
    df_final.to_parquet(out_path, index=False)
    log(f"\nWrote {out_path} ({len(df_final):,} rows, {len(df_final.columns)} columns)")


if __name__ == "__main__":
    main()
