#!/usr/bin/env python3
"""
Step 5 (Stage 5): decision engine -- turn calibrated probabilities into the
final matching_results.tsv + candidate_pairs.tsv submission files.

Steps, per the hardware-aware plan doc's Stage 5 spec:
  1. Global one-parent-per-candidate constraint: if an S2/S3 record was
     surfaced as a candidate for more than one S1 entity, keep only the
     highest-scoring (S1, candidate) assignment and drop the rest. One
     groupby, dataset-wide. Does NOT limit how many matches one S1 can have.
  2. Flat-threshold decision rule using Stage 4's best threshold (found via
     grid search on a genuinely held-out set in stage4_train_model.py).
  3. (Monte Carlo expected-F0.5 prefix selection -- deferred; the flat
     threshold already scores 0.9481 macro F_0.5 on held-out data, which is
     strong, and the plan's own fallback-ladder principle says ship the
     simpler working version first. Left as a documented next step, not
     implemented in this pass.)
  4. Write output/matching_results.tsv and output/candidate_pairs.tsv in
     the exact required shape, reusing finalize_candidate_pairs()
     (candidate_pairs.tsv directly; matching_results.tsv via the same
     function + a column rename, since that function's output column name
     is hardcoded to candidate_entity_ids -- documented in
     schema_contracts.py).
  5. Enforce the README's explicit rule that every matched id in
     matching_results.tsv must also appear in that S1's candidate list in
     candidate_pairs.tsv -- true by construction here since matches are a
     filtered subset of the same candidate pairs, but verified explicitly
     anyway rather than assumed.
  6. Run the real validator against both output files.

Input:
  pipeline_output/stage4_probabilities.parquet (from stage4_train_model.py)
  pipeline_output/stage4_model_report.json (for the chosen threshold)
  src/sampled_data/sample_source1.tsv (required S1 ids for this sample-scale run)
  src/sampled_data/sample_ground_truth.tsv (to report the REAL achieved
    macro F_0.5 on this run, for the methodology writeup)

Output:
  pipeline_output/output/matching_results.tsv
  pipeline_output/output/candidate_pairs.tsv

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/stage5_decision_engine.py
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
from finalize_candidate_pairs import finalize_candidate_pairs

REPO_ROOT = os.path.abspath(os.path.join(audit.STUDENT_RESOURCE, ".."))
SAMPLED_DATA_DIR = os.path.join(REPO_ROOT, "src", "sampled_data")
PIPELINE_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline_output")
)
OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "output")


def log(msg):
    print(msg, flush=True)


def f_beta_score(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def enforce_one_parent_per_candidate(df, score_col="calibrated_probability"):
    """For every candidate_entity_id claimed by more than one S1, keep only
    the highest-scoring (S1, candidate) row; drop the rest. Does not touch
    how many candidates a given S1 can keep."""
    before = len(df)
    df_sorted = df.sort_values(score_col, ascending=False)
    df_dedup = df_sorted.drop_duplicates(subset="candidate_entity_id", keep="first")
    after = len(df_dedup)
    n_dropped = before - after
    log(f"One-parent-per-candidate constraint: dropped {n_dropped:,} of "
        f"{before:,} rows ({100*n_dropped/before:.2f}%) where a candidate "
        f"was claimed by multiple S1 entities.")
    return df_dedup


def macro_f05_real(matches_df, gt_map, all_s1_ids):
    """Real macro F_0.5 against ACTUAL ground truth (not the held-out proxy
    used during threshold selection in stage4) -- this is the true
    challenge-equivalent score for this sample-scale run."""
    scores = []
    matches_by_s1 = matches_df.groupby("source1_entity_id")["matched_entity_ids"].first()
    for s1_id in all_s1_ids:
        true_matches = gt_map.get(s1_id, set())
        predicted_str = matches_by_s1.get(s1_id, "")
        predicted = set(predicted_str.split(",")) if predicted_str else set()

        if not true_matches and not predicted:
            scores.append(1.0)
            continue
        if not predicted:
            scores.append(0.0)
            continue

        n_correct = len(true_matches & predicted)
        precision = n_correct / len(predicted) if predicted else 0.0
        recall = n_correct / len(true_matches) if true_matches else 0.0
        scores.append(f_beta_score(precision, recall, beta=0.5))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    probs_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage4_probabilities.parquet")
    report_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage4_model_report.json")
    s1_path = os.path.join(SAMPLED_DATA_DIR, "sample_source1.tsv")
    gt_path = os.path.join(SAMPLED_DATA_DIR, "sample_ground_truth.tsv")

    for p in (probs_path, report_path, s1_path, gt_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected input not found: {p} -- run earlier stages first.")

    log(f"Loading calibrated probabilities from {probs_path}...")
    probs = pd.read_parquet(probs_path)
    log(f"  {len(probs):,} candidate pairs")

    with open(report_path, encoding="utf-8") as f:
        model_report = json.load(f)
    threshold = model_report["best_flat_threshold"]
    log(f"Using flat threshold from Stage 4's held-out sweep: {threshold} "
        f"(held-out macro F_0.5 at this threshold: "
        f"{model_report['best_flat_threshold_macro_f0.5']})")

    log("")
    log("=== Step A: global one-parent-per-candidate constraint ===")
    deduped = enforce_one_parent_per_candidate(probs)

    log("")
    log("=== Step B: applying flat threshold decision rule ===")
    matched = deduped[deduped["calibrated_probability"] >= threshold].copy()
    log(f"  {len(matched):,} of {len(deduped):,} deduped candidate pairs "
        f"pass the threshold and become final matches")

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    required_ids = s1_df["entity_id"].tolist()

    log("")
    log("=== Step C: writing output files ===")

    # candidate_pairs.tsv: the FULL candidate set (all pairs the model
    # scored), not just the ones that passed the threshold -- per the
    # README, this must be "the exact set of records you feed into your
    # matching model for inference," i.e. every pair Stage 4 scored.
    candidate_pairs_final = finalize_candidate_pairs(
        probs, required_ids, s1_col="s1_entity_id", cid_col="candidate_entity_id"
    )
    candidate_pairs_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    candidate_pairs_final.to_csv(candidate_pairs_path, sep="\t", index=False)
    log(f"  wrote {candidate_pairs_path} ({len(candidate_pairs_final):,} rows)")

    # matching_results.tsv: only the thresholded matches, same
    # finalize_candidate_pairs() function (its output column name is
    # hardcoded to candidate_entity_ids -- see schema_contracts.py -- so we
    # rename it afterward rather than fork the function).
    matching_results_final = finalize_candidate_pairs(
        matched, required_ids, s1_col="s1_entity_id", cid_col="candidate_entity_id"
    )
    matching_results_final = matching_results_final.rename(
        columns={"candidate_entity_ids": "matched_entity_ids"}
    )
    matching_results_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    matching_results_final.to_csv(matching_results_path, sep="\t", index=False)
    n_with_matches = (matching_results_final["matched_entity_ids"] != "").sum()
    log(f"  wrote {matching_results_path} ({len(matching_results_final):,} rows, "
        f"{n_with_matches:,} with >=1 match, "
        f"{len(matching_results_final) - n_with_matches:,} predicted singletons)")

    log("")
    log("=== Step D: verifying matched ids are a subset of candidate ids ===")
    cand_by_s1 = candidate_pairs_final.set_index("source1_entity_id")["candidate_entity_ids"]
    violations = 0
    for s1_id, matched_str in zip(matching_results_final["source1_entity_id"],
                                   matching_results_final["matched_entity_ids"]):
        if not matched_str:
            continue
        matched_set = set(matched_str.split(","))
        cand_set = set(cand_by_s1.get(s1_id, "").split(",")) if cand_by_s1.get(s1_id, "") else set()
        if not matched_set.issubset(cand_set):
            violations += 1
    if violations:
        raise ValueError(f"{violations} S1 entities have matched ids NOT in "
                          f"their candidate list -- this should be impossible "
                          f"by construction, investigate.")
    log("  OK -- every matched id is a subset of that S1's candidate ids "
        "(verified explicitly, not just assumed).")

    log("")
    log("=== Step E: validating against utils/validate_submission.py ===")
    utils_dir = os.path.join(audit.STUDENT_RESOURCE, "utils")
    sys.path.insert(0, utils_dir)
    import validate_submission as validator

    temp_test_dir = os.path.join(PIPELINE_OUTPUT_DIR, "_temp_test_dir_for_validation")
    os.makedirs(temp_test_dir, exist_ok=True)
    temp_s1_path = os.path.join(temp_test_dir, "test_source1.tsv")
    s1_df.to_csv(temp_s1_path, sep="\t", index=False)

    errors, warnings = validator.validate(matching_results_path, candidate_pairs_path, temp_test_dir)
    for w in warnings:
        log(f"  WARNING: {w}")
    if errors:
        log(f"  FAIL -- {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            log(f"    {i}. {e}")
        raise SystemExit(1)
    log("  PASS -- both output files are structurally valid per the "
        "challenge's own validator.")
    os.remove(temp_s1_path)
    os.rmdir(temp_test_dir)

    log("")
    log("=== Step F: real macro F_0.5 against actual ground truth ===")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map = {}
    for s1, matched_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        gt_map[s1] = set(matched_str.split(",")) if matched_str else set()

    real_f05 = macro_f05_real(matching_results_final, gt_map, required_ids)
    log(f"REAL macro F_0.5 on the full 1,000-S1 sample "
        f"(train+calibration+held-out combined, since matching_results.tsv "
        f"covers all 1,000 S1s): {real_f05:.4f}")
    log("")
    log("NOTE: this number is optimistic relative to a true test-set score, "
        "since ~90% of these S1 entities' pairs were used to TRAIN the "
        "model (only the held-out scoring fold from stage4 -- 100 S1 "
        "entities -- was genuinely unseen). The held-out-only macro F_0.5 "
        "reported in stage4_model_report.json (0.9481) is the more honest "
        "estimate of real-world performance.")

    summary = {
        "threshold_used": threshold,
        "n_candidate_pairs_scored": int(len(probs)),
        "n_pairs_after_one_parent_constraint": int(len(deduped)),
        "n_final_matches": int(len(matched)),
        "n_predicted_singletons": int(len(matching_results_final) - n_with_matches),
        "real_macro_f05_full_sample_optimistic": round(real_f05, 4),
        "held_out_macro_f05_honest_estimate": model_report["best_flat_threshold_macro_f0.5"],
        "validator_result": "PASS",
    }
    summary_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
