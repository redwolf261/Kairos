#!/usr/bin/env python3
"""
Step 4 (Stage 4): train a LightGBM pairwise classifier and calibrate its
output probabilities.

Critical correctness point (per the hardware-aware plan doc's own warning):
GroupKFold BY S1 ID -- if a single S1 entity's candidate pairs were split
across train and validation folds, the model could effectively memorize
that S1's true match from training and "predict" it in validation, making
the validation score silently lie. Grouping by s1_entity_id prevents that:
every pair belonging to a given S1 entity stays in the same fold.

Calibration: LightGBM's raw predict_proba output is not necessarily a
well-calibrated probability. Stage 5's threshold search and (later) Monte
Carlo expected-F0.5 selection both need real probabilities, so this script
applies post-hoc isotonic calibration (sklearn CalibratedClassifierCV
wrapping a FrozenEstimator around the already-fitted model, then fit
against a held-out calibration fold -- this is the modern replacement for
the removed cv="prefit" API, see the inline comment where it's used).

Input: pipeline_output/stage3_features.parquet
Output:
  pipeline_output/stage4_probabilities.parquet -- calibrated probability
    for every (s1_entity_id, candidate_entity_id) pair, per
    schema_contracts.py's STAGE4_PROBABILITY_FORMAT_DRAFT shape
  pipeline_output/stage4_model_report.json -- validation metrics (AUC,
    F0.5 at a few thresholds, calibration curve summary) for the
    Documentation_template.md writeup later

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/stage4_train_model.py
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import PIPELINE_OUTPUT_DIR, NON_FEATURE_COLS, macro_f05_per_entity, log

# Tunable knobs, overridable via env var without editing code -- same
# pattern as audit.py's SAMPLE_ROWS/AUDIT_WORKERS/BLOCKING_SAMPLE_FRAC.
N_SPLITS = int(os.environ.get("STAGE4_N_SPLITS", "5"))
RANDOM_SEED = int(os.environ.get("STAGE4_RANDOM_SEED", "42"))
N_ESTIMATORS = int(os.environ.get("STAGE4_N_ESTIMATORS", "300"))
LEARNING_RATE = float(os.environ.get("STAGE4_LEARNING_RATE", "0.05"))
NUM_LEAVES = int(os.environ.get("STAGE4_NUM_LEAVES", "31"))
MIN_CHILD_SAMPLES = int(os.environ.get("STAGE4_MIN_CHILD_SAMPLES", "20"))
THRESHOLD_SWEEP_START = float(os.environ.get("STAGE4_THRESHOLD_SWEEP_START", "0.1"))
THRESHOLD_SWEEP_STOP = float(os.environ.get("STAGE4_THRESHOLD_SWEEP_STOP", "0.95"))
THRESHOLD_SWEEP_STEP = float(os.environ.get("STAGE4_THRESHOLD_SWEEP_STEP", "0.05"))


def macro_f05_at_threshold(df, threshold):
    """Wraps the shared macro_f05_per_entity() scorer: builds the
    per-entity true/predicted id SETS this specific call needs (predicted
    membership is itself derived from `threshold`, so it can't be
    precomputed once and passed in) and delegates the actual F_0.5 math to
    pipeline_common.py -- the same implementation stage5_decision_engine.py
    uses for its real-ground-truth check, not a second copy of it."""
    true_sets, predicted_sets = {}, {}
    for s1_id, group in df.groupby("s1_entity_id"):
        true_sets[s1_id] = set(group.loc[group["label"] == 1, "candidate_entity_id"])
        predicted_sets[s1_id] = set(
            group.loc[group["calibrated_probability"] >= threshold, "candidate_entity_id"]
        )
    return macro_f05_per_entity(true_sets, predicted_sets, df["s1_entity_id"].unique())


def main():
    in_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage3_features.parquet")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{in_path} not found -- run stage3_features.py first.")

    log(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)
    log(f"  {len(df):,} rows")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # cast bool feature columns to int for LightGBM
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    log(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["s1_entity_id"].to_numpy()

    log(f"GroupKFold({N_SPLITS}) split by s1_entity_id (prevents an S1's "
        f"pairs leaking across train/validation)...")
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_indices = list(gkf.split(X, y, groups))

    # use the LAST fold as a held-out validation+calibration set; train on
    # the rest. (A full n-fold CV loop would be more thorough but this is
    # the sample-scale pass -- the full-scale phase can extend this to a
    # proper cross-validated ensemble if needed.)
    train_idx, val_idx = fold_indices[-1]
    # split val further into a calibration half and a final-scoring half,
    # still respecting group membership, so calibration doesn't leak into
    # the reported validation score either.
    val_groups = groups[val_idx]
    unique_val_groups = np.unique(val_groups)
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_val_groups)
    half = len(unique_val_groups) // 2
    calib_groups = set(unique_val_groups[:half])
    calib_mask = np.isin(val_groups, list(calib_groups))
    calib_idx = val_idx[calib_mask]
    score_idx = val_idx[~calib_mask]

    log(f"  train: {len(train_idx):,} pairs ({len(np.unique(groups[train_idx])):,} S1 entities)")
    log(f"  calibration: {len(calib_idx):,} pairs ({len(calib_groups):,} S1 entities)")
    log(f"  held-out scoring: {len(score_idx):,} pairs "
        f"({len(unique_val_groups) - half:,} S1 entities)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]
    X_score, y_score = X[score_idx], y[score_idx]

    log("Training LightGBM...")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        # positive class is ~4% of pairs -- let LightGBM weight accordingly
        # rather than resampling, since the sample is small enough that
        # resampling would throw away too much signal
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    log(f"  trained in {round(time.time()-t0,1)}s")

    raw_val_auc = roc_auc_score(y_score, model.predict_proba(X_score)[:, 1])
    log(f"Raw (uncalibrated) AUC on held-out scoring set: {raw_val_auc:.4f}")

    log("Calibrating (isotonic, prefit against the calibration fold)...")
    # cv="prefit" was removed in newer scikit-learn (deprecated 1.6, gone in
    # 1.7+) -- FrozenEstimator is the current replacement: it wraps an
    # already-fitted estimator so CalibratedClassifierCV treats it as fixed
    # and only fits the calibration mapping on X_calib/y_calib, exactly like
    # the old cv="prefit" behavior.
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    calibrated_probs_score = calibrated_model.predict_proba(X_score)[:, 1]
    calibrated_auc = roc_auc_score(y_score, calibrated_probs_score)
    log(f"Calibrated AUC on held-out scoring set: {calibrated_auc:.4f} "
        f"(AUC is threshold-invariant so this should be very close to the "
        f"raw AUC -- calibration reshapes probabilities, not ranking)")

    if not (0 <= calibrated_probs_score.min() and calibrated_probs_score.max() <= 1):
        raise ValueError("Calibrated probabilities outside [0, 1] -- calibration bug.")
    log(f"Calibrated probability range: "
        f"[{calibrated_probs_score.min():.4f}, {calibrated_probs_score.max():.4f}] -- OK")

    # --- threshold sweep on the held-out scoring set, using the real
    # challenge scoring definition (macro F_0.5 per S1 entity)
    score_df = df.iloc[score_idx][["s1_entity_id", "candidate_entity_id", "label"]].copy()
    score_df["calibrated_probability"] = calibrated_probs_score

    log("")
    log("=== Threshold sweep (macro F_0.5, held-out scoring set) ===")
    thresholds = np.arange(THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_STOP, THRESHOLD_SWEEP_STEP)
    threshold_results = []
    for t in thresholds:
        f05 = macro_f05_at_threshold(score_df, t)
        threshold_results.append((round(float(t), 2), f05))
        log(f"  threshold={t:.2f}  macro_F0.5={f05:.4f}")

    best_threshold, best_f05 = max(threshold_results, key=lambda x: x[1])
    log(f"\nBest flat threshold on held-out scoring set: {best_threshold} "
        f"(macro F_0.5 = {best_f05:.4f})")

    # --- score EVERY pair in the full dataset with the final calibrated
    # model (trained on train_idx, calibrated on calib_idx) -- this is what
    # Stage 5 needs: a probability for every candidate pair, not just the
    # held-out subset.
    log("")
    log("Scoring all pairs with the final calibrated model...")
    all_probs = calibrated_model.predict_proba(X)[:, 1]

    output_df = df[["s1_entity_id", "candidate_entity_id"]].copy()
    output_df["raw_score"] = model.predict_proba(X)[:, 1]
    output_df["calibrated_probability"] = all_probs

    out_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage4_probabilities.parquet")
    output_df.to_parquet(out_path, index=False)
    log(f"Wrote {out_path} ({len(output_df):,} rows)")

    report = {
        "n_training_pairs": int(len(train_idx)),
        "n_calibration_pairs": int(len(calib_idx)),
        "n_held_out_scoring_pairs": int(len(score_idx)),
        "feature_columns": feature_cols,
        "raw_auc_held_out": round(float(raw_val_auc), 4),
        "calibrated_auc_held_out": round(float(calibrated_auc), 4),
        "threshold_sweep": [{"threshold": t, "macro_f0.5": f} for t, f in threshold_results],
        "best_flat_threshold": best_threshold,
        "best_flat_threshold_macro_f0.5": round(float(best_f05), 4),
        "calibrated_probability_range": [
            round(float(calibrated_probs_score.min()), 4),
            round(float(calibrated_probs_score.max()), 4),
        ],
    }
    report_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage4_model_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
