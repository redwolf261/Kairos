#!/usr/bin/env python3
"""
Fair, apples-to-apples comparison of the teammate's 5 model types
(logistic_regression, random_forest, extra_trees, hist_gradient_boosting,
rbf_svm -- src/model_evaluation.ipynb) against this pipeline's LightGBM
(stage4_train_model.py), on IDENTICAL data and IDENTICAL scoring.

Why this script exists: the teammate's reported numbers (e.g. extra_trees
precision=0.9982 recall=0.9862 F0.5=0.9958) are NOT comparable to this
pipeline's reported macro F_0.5 = 0.9836, for two independent reasons:

  1. DIFFERENT METRIC. Their F0.5 is sklearn's fbeta_score(beta=0.5) over
     individual PAIR predictions (every candidate pair is one classification
     example, scored as if they were independent). The challenge's actual
     scoring metric (student_resource/README.md) is macro-averaged F_0.5
     PER S1 ENTITY -- computed from that entity's full predicted match set
     vs. its true match set, then averaged across every S1 entity
     (singletons included, where a correct empty prediction scores 1.0 and
     any false positive on a singleton scores 0.0). These are genuinely
     different numbers for the same predictions, not just a relabeling.

  2. DIFFERENT DATA. build_training_dataset.py's build_dataset() unions
     EVERY ground-truth true pair into the candidate pool before training/
     testing, even pairs the actual blocking output never surfaced. That
     makes their train/test set easier than a real deployment scenario,
     where you only ever see what blocking found. This pipeline's
     stage3_features.parquet (produced by build_training_pairs.py) labels
     ONLY pairs that survived real blocking -- no injection.

This script controls for BOTH: same 5 model types (mirroring
model_evaluation.ipynb's exact hyperparameters), trained and evaluated on
THIS pipeline's real (non-inflated) stage3_features.parquet, with the same
GroupKFold-by-S1-entity split stage4_train_model.py already uses, scored
with the same macro_f05_per_entity() function stage4/stage5 use -- so the
resulting numbers are directly comparable to stage4_model_report.json's
0.9836, and to each other.

Input: pipeline_output/stage3_features.parquet (from stage3_features.py)
Output: pipeline_output/fair_model_comparison.json

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/fair_model_comparison.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import PIPELINE_OUTPUT_DIR, NON_FEATURE_COLS, macro_f05_per_entity, log

RANDOM_SEED = int(os.environ.get("COMPARISON_RANDOM_SEED", "42"))
N_SPLITS = int(os.environ.get("COMPARISON_N_SPLITS", "5"))

# Mirrors model_evaluation.ipynb's exact hyperparameters (per the earlier
# extraction), so any performance difference reflects the data/metric
# fixes this script applies, not a hyperparameter change.
MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_SEED),
    "rbf_svm": lambda: SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
}
# these two need scaled features (matches model_evaluation.ipynb's approach)
NEEDS_SCALING = {"logistic_regression", "rbf_svm"}


def macro_f05_at_threshold(df, threshold, score_col="probability"):
    """Same per-entity F_0.5 computation stage4_train_model.py uses,
    generalized to take an arbitrary score column name so it works for
    any of the 6 models compared here (LightGBM's own version of this
    function lives in stage4_train_model.py; this is intentionally a
    separate, self-contained copy rather than an import, since this
    script computes it for 5 DIFFERENT models' score columns in a loop
    rather than once for a single trained model -- see pipeline_common.py
    for the single-source-of-truth math this wraps)."""
    true_sets, predicted_sets = {}, {}
    for s1_id, group in df.groupby("s1_entity_id"):
        true_sets[s1_id] = set(group.loc[group["label"] == 1, "candidate_entity_id"])
        predicted_sets[s1_id] = set(
            group.loc[group[score_col] >= threshold, "candidate_entity_id"]
        )
    return macro_f05_per_entity(true_sets, predicted_sets, df["s1_entity_id"].unique())


def main():
    in_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage3_features.parquet")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{in_path} not found -- run stage3_features.py first.")

    log(f"Loading {in_path} (real blocking output, no ground-truth injection)...")
    df = pd.read_parquet(in_path)
    log(f"  {len(df):,} rows, {df['label'].sum():,} positive ({100*df['label'].mean():.2f}%)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    log(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["s1_entity_id"].to_numpy()

    # SAME split methodology as stage4_train_model.py: GroupKFold by S1
    # entity, last fold held out, further split into calibration/scoring
    # halves by group -- so every model here sees the exact same
    # train/scoring partition LightGBM was evaluated on.
    log(f"GroupKFold({N_SPLITS}) split by s1_entity_id (same partition stage4 used)...")
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_indices = list(gkf.split(X, y, groups))
    train_idx, val_idx = fold_indices[-1]

    val_groups = groups[val_idx]
    unique_val_groups = np.unique(val_groups)
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_val_groups)
    half = len(unique_val_groups) // 2
    calib_groups = set(unique_val_groups[:half])
    calib_mask = np.isin(val_groups, list(calib_groups))
    calib_idx = val_idx[calib_mask]
    score_idx = val_idx[~calib_mask]

    log(f"  train: {len(train_idx):,} pairs, held-out scoring: {len(score_idx):,} pairs "
        f"({len(unique_val_groups) - half:,} S1 entities)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_score, y_score = X[score_idx], y[score_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_score_scaled = scaler.transform(X_score)

    score_df_base = df.iloc[score_idx][["s1_entity_id", "candidate_entity_id", "label"]].reset_index(drop=True)

    results = {}
    for name, factory in MODEL_FACTORIES.items():
        log(f"\n=== {name} ===")
        t0 = time.time()
        model = factory()
        Xtr = X_train_scaled if name in NEEDS_SCALING else X_train
        Xsc = X_score_scaled if name in NEEDS_SCALING else X_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SVC(probability=True) deprecation noise, not our concern here
            model.fit(Xtr, y_train)
        train_seconds = round(time.time() - t0, 1)

        probs = model.predict_proba(Xsc)[:, 1]
        score_df = score_df_base.copy()
        score_df["probability"] = probs

        # per-PAIR fbeta (the metric the teammate's notebook reports) --
        # included for direct comparability against their own numbers
        from sklearn.metrics import fbeta_score, roc_auc_score
        pair_preds_at_05 = (probs >= 0.5).astype(int)
        per_pair_f05_at_default_threshold = fbeta_score(y_score, pair_preds_at_05, beta=0.5, zero_division=0)
        auc = roc_auc_score(y_score, probs) if len(set(y_score)) > 1 else None

        # the metric that actually matters: macro F_0.5 PER S1 ENTITY,
        # swept over thresholds exactly like stage4_train_model.py does
        thresholds = np.arange(0.1, 0.95, 0.05)
        threshold_results = [(round(float(t), 2), macro_f05_at_threshold(score_df, t)) for t in thresholds]
        best_threshold, best_macro_f05 = max(threshold_results, key=lambda x: x[1])

        log(f"  trained in {train_seconds}s")
        log(f"  per-pair F0.5 @ threshold=0.5 (the teammate's metric): {per_pair_f05_at_default_threshold:.4f}")
        log(f"  per-pair AUC: {auc:.4f}" if auc is not None else "  per-pair AUC: n/a (single class in scoring set)")
        log(f"  REAL macro F_0.5 per S1 entity (the challenge's metric), "
            f"best threshold={best_threshold}: {best_macro_f05:.4f}")

        results[name] = {
            "train_seconds": train_seconds,
            "per_pair_fbeta05_at_threshold_0.5": round(float(per_pair_f05_at_default_threshold), 4),
            "per_pair_auc": round(float(auc), 4) if auc is not None else None,
            "best_threshold": best_threshold,
            "macro_f05_per_entity": round(float(best_macro_f05), 4),
            "threshold_sweep": [{"threshold": t, "macro_f0.5": f} for t, f in threshold_results],
        }

    log("\n" + "=" * 70)
    log("SUMMARY -- macro F_0.5 per S1 entity (the actual challenge metric),")
    log("all models on IDENTICAL real (non-inflated) data and split:")
    log("=" * 70)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["macro_f05_per_entity"], reverse=True)
    for name, r in ranked:
        log(f"  {name:24s}  macro_F0.5={r['macro_f05_per_entity']:.4f}  "
            f"(per-pair F0.5={r['per_pair_fbeta05_at_threshold_0.5']:.4f}, "
            f"threshold={r['best_threshold']})")

    log(f"\n  {'lightgbm (stage4)':24s}  macro_F0.5=0.9836  <- from stage4_model_report.json, "
        f"same GroupKFold split, for reference")

    out_path = os.path.join(PIPELINE_OUTPUT_DIR, "fair_model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "All models trained/scored on IDENTICAL data (stage3_features.parquet, "
                    "real blocking output, no ground-truth injection) and IDENTICAL "
                    "GroupKFold-by-s1_entity_id split. macro_f05_per_entity is the "
                    "challenge's actual scoring metric; per_pair_fbeta05 is what "
                    "model_evaluation.ipynb reports (NOT the same metric, kept here "
                    "for direct comparison to the teammate's own reported numbers).",
            "lightgbm_reference_macro_f05": 0.9836,
            "models": results,
        }, f, indent=2)
    log(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
