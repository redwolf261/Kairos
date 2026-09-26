#!/usr/bin/env python3
"""
Person D (GPU owner / integration owner) -- smoke-test harness.

Per the plan: "Do a full, ugly, end-to-end smoke test as early as anyone can
force one through... four people's code failing to connect is the single
most common way hackathon teams lose time, and it's much cheaper to find at
hour 8 than hour 20." This is that check, runnable at any point with
whatever real files exist and stub data for whatever doesn't yet -- so it's
useful from hour 1, not just once every stage is finished.

Two things, layered:

  1. FINAL SUBMISSION VALIDATION -- a thin wrapper around the challenge's own
     utils/validate_submission.py, so it's trivial to run repeatedly during
     development instead of remembering the right flags each time. This part
     needs NOTHING from teammates: only the challenge's own dataset/test/
     files, which are already local to everyone.

  2. INTERMEDIATE SCHEMA CHECKS -- checks whatever intermediate stage files
     already exist (Stage 1 normalized output, Stage 2's candidate_pairs,
     Stage 3's feature files, Stage 4's probability file) against the DRAFT
     contract in schema_contracts.py. Each check is independent and skips
     cleanly if that stage's file doesn't exist yet -- so this is safe to run
     at hour 1 (everything skipped) or hour 20 (everything checked), and
     anywhere in between as teammates' files land.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/smoke_test.py [options]

Options:
    --matching PATH       path to matching_results.tsv (default: output/matching_results.tsv)
    --candidate PATH      path to candidate_pairs.tsv (default: output/candidate_pairs.tsv)
    --stage1 PATH         path to a Stage 1 normalized-output file to check (optional)
    --stage2-flat PATH    path to an intermediate flat candidate-pair file, e.g.
                          blocking.ipynb's working format, BEFORE it's fused
                          into candidate_pairs.tsv (optional)
    --stage3 PATH         path to a Stage 3 feature Parquet/TSV file (optional)
    --stage4 PATH         path to a Stage 4 probability file (optional)
    --test-dir PATH       dataset/test directory (default: dataset/test)

Exit code 0 = everything checked passed (or was skipped because the file
doesn't exist yet). Exit code 1 = at least one real problem found.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema_contracts as sc

VALIDATOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "utils", "validate_submission.py"
)
VALIDATOR_PATH = os.path.normpath(VALIDATOR_PATH)


def log(msg):
    print(msg, flush=True)


def section(title):
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


# ---------------------------------------------------------------------------
# 1. Final submission validation -- wraps the challenge's own validator
# ---------------------------------------------------------------------------

def run_final_validator(matching_path, candidate_path, test_dir):
    section("1. FINAL SUBMISSION VALIDATION (wraps utils/validate_submission.py)")

    if not os.path.isfile(matching_path):
        log(f"SKIP -- {matching_path} does not exist yet. This check runs "
            f"once Stage 5 produces a real matching_results.tsv.")
        return None  # skipped, not a failure

    if not os.path.isfile(VALIDATOR_PATH):
        log(f"WARNING -- could not find the challenge's validator at "
            f"{VALIDATOR_PATH}. Skipping this check.")
        return None

    sys.path.insert(0, os.path.dirname(VALIDATOR_PATH))
    import validate_submission as validator

    log(f"Validating: {matching_path}")
    if os.path.isfile(candidate_path):
        log(f"       and: {candidate_path}")
    else:
        log(f"(candidate file {candidate_path} not found -- validator will "
            f"warn but not fail on this)")

    errors, warnings = validator.validate(matching_path, candidate_path, test_dir)

    for w in warnings:
        log(f"WARNING: {w}")
    if errors:
        log(f"FAIL -- {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
        return False
    log("PASS -- matching_results.tsv is safe to submit.")
    return True


# ---------------------------------------------------------------------------
# 2. Intermediate schema checks
# ---------------------------------------------------------------------------

def _read_any(path):
    """Read a .tsv or .parquet transparently, since different stages may
    land in either format depending on who built them."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def check_stage1(path):
    section("2a. STAGE 1 -- normalized-field output")
    if not path:
        log("SKIP -- no --stage1 path given.")
        return None
    if not os.path.isfile(path):
        log(f"SKIP -- {path} does not exist yet.")
        return None

    df = _read_any(path)
    log(f"Loaded {path}: {len(df):,} rows, columns: {list(df.columns)}")

    required = sc.STAGE1_NORMALIZED_FIELDS["required_columns"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        log(f"FAIL -- missing required columns: {missing_required}")
        return False

    known_normalized = set(sc.STAGE1_NORMALIZED_FIELDS["draft_normalized_columns"])
    present_normalized = [c for c in df.columns if c in known_normalized]
    unknown_extra = [
        c for c in df.columns
        if c not in required and c not in known_normalized
    ]
    log(f"Required columns present: OK ({required})")
    log(f"Recognized normalized columns present: {present_normalized or '(none)'}")
    if unknown_extra:
        log(f"NOTE -- columns not in the draft contract (not necessarily "
            f"wrong, just undocumented -- update schema_contracts.py if "
            f"these are intentional): {unknown_extra}")

    if not present_normalized:
        log("FAIL -- no recognized normalized-name/address column found at "
            "all. Downstream blocking code has nothing to join on.")
        return False

    log("PASS -- Stage 1 output has the required raw columns plus at least "
        "one recognized normalized column.")
    return True


def check_stage2_flat(path):
    section("2b. STAGE 2 -- intermediate flat candidate-pair file")
    if not path:
        log("SKIP -- no --stage2-flat path given.")
        return None
    if not os.path.isfile(path):
        log(f"SKIP -- {path} does not exist yet.")
        return None

    df = _read_any(path)
    log(f"Loaded {path}: {len(df):,} rows, columns: {list(df.columns)}")

    intermediate = sc.CANDIDATE_PAIRS_INTERMEDIATE_FORMAT_SEEN_IN_REPO["columns"]
    final = sc.CANDIDATE_PAIRS_FINAL_FORMAT["columns"]

    if list(df.columns[:2]) == intermediate:
        log(f"This matches the FLAT intermediate format ({intermediate}) "
            f"seen in src/blocking.ipynb -- one row per (S1, candidate) pair.")
        log("REMINDER: this is NOT the final candidate_pairs.tsv shape. "
            "Run it through finalize_candidate_pairs.py (same directory) to "
            f"convert to the required {final} shape before packaging.")
        n_s1 = df[intermediate[0]].nunique()
        n_pairs = len(df)
        log(f"Distinct S1 entities represented: {n_s1:,}. "
            f"Total pairs: {n_pairs:,}. "
            f"Avg candidates/S1 (in this file, not necessarily the final "
            f"fused count): {n_pairs/n_s1:.1f}" if n_s1 else "")
        dupes = df.duplicated(subset=intermediate).sum()
        if dupes:
            log(f"NOTE -- {dupes:,} duplicate (s1, candidate) rows found "
                f"(harmless if a later drop_duplicates() step already "
                f"exists, but check).")
        log("PASS (as an intermediate file) -- structurally sound flat pair table.")
        return True

    if list(df.columns[:2]) == final:
        log(f"This matches the FINAL candidate_pairs.tsv shape already "
            f"({final}) -- one row per S1 entity, comma-joined candidates.")
        empty_lists = (df[final[1]] == "").sum()
        log(f"{len(df):,} S1 rows, {empty_lists:,} with an empty candidate list.")
        log("PASS -- already in final shape; run the full validator (check 1) "
            "against it once it's placed as output/candidate_pairs.tsv.")
        return True

    log(f"FAIL -- columns {list(df.columns[:2])} match neither the known "
        f"intermediate format {intermediate} nor the final format {final}. "
        f"This is exactly the kind of schema drift the smoke test exists to "
        f"catch -- reconcile with schema_contracts.py or update the contract "
        f"if this is an intentional new format.")
    return False


def check_stage3(path):
    section("2c. STAGE 3 -- feature file")
    if not path:
        log("SKIP -- no --stage3 path given.")
        return None
    if not os.path.isfile(path):
        log(f"SKIP -- {path} does not exist yet.")
        return None

    df = _read_any(path)
    log(f"Loaded {path}: {len(df):,} rows, columns: {list(df.columns)}")

    required = sc.STAGE3_FEATURE_FILE_DRAFT["required_columns"]
    # "label" is optional (only present in training files) -- check the join
    # keys, warn (don't fail) on label's absence.
    join_keys = [c for c in required if c != "label"]
    missing = [c for c in join_keys if c not in df.columns]
    if missing:
        log(f"FAIL -- missing required join-key columns: {missing}")
        return False

    if "label" not in df.columns:
        log("NOTE -- no 'label' column: OK if this is an inference feature "
            "file, NOT ok if this is meant to be a training file.")

    feature_cols = [c for c in df.columns if c not in required]
    log(f"Feature columns present ({len(feature_cols)}): {feature_cols}")

    known_features = set(sc.STAGE3_FEATURE_FILE_DRAFT["feature_columns_draft"])
    unknown = [c for c in feature_cols if c not in known_features]
    if unknown:
        log(f"NOTE -- feature columns not in the draft list (fine if "
            f"intentional, update schema_contracts.py to match reality): "
            f"{unknown}")

    log("PASS -- required join-key columns present.")
    return True


def check_stage4(path):
    section("2d. STAGE 4 -- calibrated probability output")
    if not path:
        log("SKIP -- no --stage4 path given.")
        return None
    if not os.path.isfile(path):
        log(f"SKIP -- {path} does not exist yet.")
        return None

    df = _read_any(path)
    log(f"Loaded {path}: {len(df):,} rows, columns: {list(df.columns)}")

    required = sc.STAGE4_PROBABILITY_FORMAT_DRAFT["required_columns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log(f"FAIL -- missing required columns: {missing}")
        return False

    prob_col = "calibrated_probability"
    probs = pd.to_numeric(df[prob_col], errors="coerce")
    out_of_range = ((probs < 0) | (probs > 1)).sum()
    nan_count = probs.isna().sum()
    if out_of_range:
        log(f"FAIL -- {out_of_range:,} rows have {prob_col} outside [0, 1] "
            f"-- looks uncalibrated or a units bug (e.g. percentages "
            f"instead of a 0-1 probability).")
        return False
    if nan_count:
        log(f"FAIL -- {nan_count:,} rows have a non-numeric/missing "
            f"{prob_col}.")
        return False

    log(f"PASS -- {prob_col} is numeric and in [0, 1] for all "
        f"{len(df):,} rows.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoke-test harness for the entity-resolution pipeline.")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--stage1", default=None)
    parser.add_argument("--stage2-flat", default=None)
    parser.add_argument("--stage3", default=None)
    parser.add_argument("--stage4", default=None)
    parser.add_argument("--test-dir", default="dataset/test")
    args = parser.parse_args()

    results = {}
    results["final_validator"] = run_final_validator(args.matching, args.candidate, args.test_dir)
    results["stage1"] = check_stage1(args.stage1)
    results["stage2_flat"] = check_stage2_flat(args.stage2_flat)
    results["stage3"] = check_stage3(args.stage3)
    results["stage4"] = check_stage4(args.stage4)

    section("SUMMARY")
    any_failure = False
    any_checked = False
    for name, result in results.items():
        if result is None:
            log(f"  {name}: SKIPPED (no file to check)")
        elif result is True:
            log(f"  {name}: PASS")
            any_checked = True
        else:
            log(f"  {name}: FAIL")
            any_checked = True
            any_failure = True

    if not any_checked:
        log("")
        log("Nothing to check yet -- no stage output files exist. This is "
            "expected at hour 0-1; re-run this as teammates' outputs land.")
        return 0

    if any_failure:
        log("")
        log("SMOKE TEST FAILED -- fix the issues above before teammates build "
            "on top of the affected stage's output.")
        return 1

    log("")
    log("SMOKE TEST PASSED -- all checked stages have consistent, "
        "contract-conforming output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
