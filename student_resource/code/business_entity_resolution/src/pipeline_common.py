#!/usr/bin/env python3
"""
Shared constants and helpers for the sample-scale pipeline scripts
(finalize_stage2_sample.py, build_training_pairs.py, stage3_features.py,
stage4_train_model.py, stage5_decision_engine.py).

Consolidates what was previously copy-pasted verbatim across those five
files: path constants, the logging helper, the join-key/non-feature column
set, the F_0.5 scoring math, and the "run the real challenge validator
against a pair of output files" routine. A change to any of these (e.g.
adding a feature column, or a schema_contracts.py rename) now only needs
to happen in one place.
"""

import os
import sys

import audit  # noqa: E402  (caller has already inserted its own dir onto sys.path)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(audit.STUDENT_RESOURCE, ".."))
BLOCKING_RESULTS_DIR = os.path.join(REPO_ROOT, "src", "blocking_results")
SAMPLED_DATA_DIR = os.path.join(REPO_ROOT, "src", "sampled_data")
PIPELINE_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline_output")
)
FINAL_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "output")

# Named input file paths -- the teammate's blocking output filenames and
# the sample dataset filenames were previously repeated as string literals
# across 3+ scripts (finalize_stage2_sample.py, build_training_pairs.py,
# stage5_decision_engine.py); a rename on the blocking side (e.g. a future
# "experiment3_*" naming) would then need updating in every one of those
# places individually, and it would be easy to miss one. One name each here
# instead. Override via env var if you're pointing at a different blocking
# run's output without renaming files.
BLOCKING_CANDIDATE_PAIRS_FILENAME = os.environ.get(
    "PIPELINE_BLOCKING_PAIRS_FILENAME", "experiment2_candidate_pairs.tsv"
)
BLOCKING_PROVENANCE_FILENAME = os.environ.get(
    "PIPELINE_BLOCKING_PROVENANCE_FILENAME", "experiment2_candidate_provenance.tsv"
)
SAMPLE_SOURCE1_PATH = os.path.join(SAMPLED_DATA_DIR, "sample_source1.tsv")
SAMPLE_SOURCE2_PATH = os.path.join(SAMPLED_DATA_DIR, "sample_source2.tsv")
SAMPLE_SOURCE3_PATH = os.path.join(SAMPLED_DATA_DIR, "sample_source3.tsv")
SAMPLE_GROUND_TRUTH_PATH = os.path.join(SAMPLED_DATA_DIR, "sample_ground_truth.tsv")
BLOCKING_CANDIDATE_PAIRS_PATH = os.path.join(BLOCKING_RESULTS_DIR, BLOCKING_CANDIDATE_PAIRS_FILENAME)
BLOCKING_PROVENANCE_PATH = os.path.join(BLOCKING_RESULTS_DIR, BLOCKING_PROVENANCE_FILENAME)


def log(msg):
    print(msg, flush=True)


def load_sample_ground_truth(path=None):
    """Parses a ground-truth TSV (source1_entity_id, matched_entity_ids)
    into a dict[str, set[str]] -- same logic as audit.load_ground_truth_map(),
    but that function is hardcoded to the FULL dataset's
    train_ground_truth.tsv path, not usable for the sample's
    sample_ground_truth.tsv. Was duplicated inline in
    build_training_pairs.py and stage5_decision_engine.py; consolidated
    here. Defaults to SAMPLE_GROUND_TRUTH_PATH if no path given."""
    import pandas as pd
    path = path or SAMPLE_GROUND_TRUTH_PATH
    gt = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for s1, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1] = set(matched.split(",")) if matched else set()
    return gt


# ---------------------------------------------------------------------------
# Column sets -- the single source of truth for "which columns are join
# keys / carried-over record data" vs. "which are actual model features."
# schema_contracts.py's STAGE3_FEATURE_FILE_DRAFT documents this same set;
# keep the two in sync if either changes.
# ---------------------------------------------------------------------------

JOIN_KEY_COLS = ["s1_entity_id", "candidate_entity_id"]

RECORD_DATA_COLS = [
    "s1_business_name", "s1_business_address", "s1_norm_name", "s1_norm_addr", "s1_country",
    "s2_business_name", "s2_business_address", "s2_norm_name", "s2_norm_addr", "s2_country",
]

PROVENANCE_COLS = ["blocking_methods", "num_blockers", "max_block_score"]

LABEL_COL = ["label"]

# what build_training_pairs.py writes -- join keys + record data + provenance + label
TRAINING_PAIRS_COLUMNS = JOIN_KEY_COLS + RECORD_DATA_COLS + PROVENANCE_COLS + LABEL_COL

# what stage4_train_model.py must exclude from the feature matrix (join
# keys, raw/normalized text, provenance strings, and the label itself --
# num_blockers/max_block_score ARE used as features under their renamed
# form route_count/max_block_score, added by stage3_features.py, so the
# ORIGINAL num_blockers/blocking_methods columns are excluded here but the
# renamed route_count is not)
NON_FEATURE_COLS = set(JOIN_KEY_COLS + RECORD_DATA_COLS + ["blocking_methods", "num_blockers"] + LABEL_COL)


# ---------------------------------------------------------------------------
# F_0.5 scoring -- the challenge's own metric (student_resource/README.md),
# used both for Stage 4's threshold sweep on held-out data and Stage 5's
# real-ground-truth check. One implementation, not two.
# ---------------------------------------------------------------------------

def f_beta_score(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f05_per_entity(true_sets_by_entity, predicted_sets_by_entity, all_entity_ids):
    """Macro-averaged F_0.5, matching the challenge's scoring definition:
    computed per entity (precision/recall over that entity's predicted vs
    true set), then averaged across ALL entity_ids -- singletons included,
    where a correct empty prediction scores 1.0 and any false-positive
    prediction on a true singleton scores 0.0.

    true_sets_by_entity / predicted_sets_by_entity: dict[str, set[str]],
    entities absent from a dict are treated as having an empty set (no
    true matches / no predictions).
    """
    scores = []
    for entity_id in all_entity_ids:
        true_set = true_sets_by_entity.get(entity_id, set())
        predicted_set = predicted_sets_by_entity.get(entity_id, set())

        if not true_set and not predicted_set:
            scores.append(1.0)
            continue
        if not predicted_set:
            scores.append(0.0)
            continue

        n_correct = len(true_set & predicted_set)
        precision = n_correct / len(predicted_set) if predicted_set else 0.0
        recall = n_correct / len(true_set) if true_set else 0.0
        scores.append(f_beta_score(precision, recall, beta=0.5))

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Validator wrapper -- runs the challenge's own utils/validate_submission.py
# against a (matching_results, candidate_pairs) pair, using a given S1 id
# list as the "required" set. Used by both finalize_stage2_sample.py (which
# only has a candidate_pairs.tsv to check, so it builds an all-empty
# placeholder matching file) and stage5_decision_engine.py (which has real
# files for both).
# ---------------------------------------------------------------------------

def run_official_validator(matching_path, candidate_path, s1_ids_df, s1_id_col="entity_id"):
    """s1_ids_df: DataFrame with a column named s1_id_col holding every
    required S1 entity_id (e.g. the sample's sample_source1.tsv, loaded
    already). Returns nothing; raises SystemExit(1) on validator failure,
    logs PASS on success. Cleans up its own temp test directory."""
    utils_dir = os.path.join(audit.STUDENT_RESOURCE, "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    import validate_submission as validator

    temp_test_dir = os.path.join(PIPELINE_OUTPUT_DIR, "_temp_test_dir_for_validation")
    os.makedirs(temp_test_dir, exist_ok=True)
    temp_s1_path = os.path.join(temp_test_dir, "test_source1.tsv")
    s1_ids_df.rename(columns={s1_id_col: "entity_id"}).to_csv(temp_s1_path, sep="\t", index=False)

    try:
        errors, warnings = validator.validate(matching_path, candidate_path, temp_test_dir)
        for w in warnings:
            log(f"  WARNING: {w}")
        if errors:
            log(f"  FAIL -- {len(errors)} issue(s):")
            for i, e in enumerate(errors, 1):
                log(f"    {i}. {e}")
            raise SystemExit(1)
        log("  PASS -- structurally valid per the challenge's own validator.")
    finally:
        if os.path.isfile(temp_s1_path):
            os.remove(temp_s1_path)
        if os.path.isdir(temp_test_dir):
            os.rmdir(temp_test_dir)
