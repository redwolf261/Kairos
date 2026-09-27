#!/usr/bin/env python3
"""
Stage 6 -- leave-one-country-out validation, per the hardware-aware plan
doc's Stage 6 spec:

    "Leave-one-country-out: train India-only -> validate as if US unseen,
    and reverse. LightGBM trains fast even on millions of rows, so this
    costs maybe 30-60 minutes total, not a real time hit -- and it's your
    only honest France proxy since training has zero French rows."

This was never built earlier -- no France-country training data exists at
all (the training set is US+India only), so this is the ONLY way to get
an honest read on how the model generalizes to a country it has never
seen a single labeled example from, which is directly relevant since the
real test set includes France.

Trains on ONE country's labeled pairs, scores on the OTHER country's
held-out pairs -- if performance holds up close to the same-country
GroupKFold number (0.9836 per stage4_model_report.json), that's evidence
the model is learning genuine name/address similarity signal rather than
country-specific quirks, and the France predictions in the real
submission are more trustworthy. If it drops sharply, that's an honest,
important caveat about the France portion of the real submission.

Input: pipeline_output/stage3_features.parquet (from stage3_features.py)
Output: pipeline_output/stage6_leave_one_country_out_report.json

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/stage6_leave_one_country_out.py
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import PIPELINE_OUTPUT_DIR, NON_FEATURE_COLS, macro_f05_per_entity, log

RANDOM_SEED = int(os.environ.get("STAGE6_RANDOM_SEED", "42"))
N_ESTIMATORS = int(os.environ.get("STAGE6_N_ESTIMATORS", "300"))
LEARNING_RATE = float(os.environ.get("STAGE6_LEARNING_RATE", "0.05"))
NUM_LEAVES = int(os.environ.get("STAGE6_NUM_LEAVES", "31"))
MIN_CHILD_SAMPLES = int(os.environ.get("STAGE6_MIN_CHILD_SAMPLES", "20"))
THRESHOLDS = np.arange(0.1, 0.95, 0.05)


def macro_f05_at_threshold(df, threshold, score_col="probability"):
    true_sets, predicted_sets = {}, {}
    for s1_id, group in df.groupby("s1_entity_id"):
        true_sets[s1_id] = set(group.loc[group["label"] == 1, "candidate_entity_id"])
        predicted_sets[s1_id] = set(
            group.loc[group[score_col] >= threshold, "candidate_entity_id"]
        )
    return macro_f05_per_entity(true_sets, predicted_sets, df["s1_entity_id"].unique())


def train_and_eval(train_df, test_df, feature_cols, train_country, test_country):
    log(f"\n=== Train on {train_country} ({len(train_df):,} pairs) -> "
        f"validate on {test_country} ({len(test_df):,} pairs) ===")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2:
        log(f"  SKIPPED: only one class present in {train_country} training data.")
        return None

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    log(f"  trained in {round(time.time()-t0,1)}s")

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else None
    log(f"  cross-country AUC: {auc:.4f}" if auc is not None else "  AUC: n/a (single class in test)")

    score_df = test_df[["s1_entity_id", "candidate_entity_id", "label"]].copy()
    score_df["probability"] = probs

    threshold_results = [(round(float(t), 2), macro_f05_at_threshold(score_df, t)) for t in THRESHOLDS]
    best_threshold, best_f05 = max(threshold_results, key=lambda x: x[1])
    log(f"  best cross-country macro F0.5: {best_f05:.4f} @ threshold={best_threshold} "
        f"(compare to same-country GroupKFold baseline in stage4_model_report.json)")

    return {
        "train_country": train_country,
        "test_country": test_country,
        "n_train_pairs": int(len(train_df)),
        "n_test_pairs": int(len(test_df)),
        "cross_country_auc": round(float(auc), 4) if auc is not None else None,
        "best_threshold": best_threshold,
        "best_macro_f05": round(float(best_f05), 4),
        "threshold_sweep": [{"threshold": t, "macro_f0.5": f} for t, f in threshold_results],
    }


def main():
    in_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage3_features.parquet")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{in_path} not found -- run stage3_features.py first.")

    log(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)
    log(f"  {len(df):,} rows")

    if "s1_country" not in df.columns:
        raise ValueError(
            "s1_country column not found in stage3_features.parquet -- "
            "cannot do a leave-one-country-out split without it."
        )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    countries_present = sorted(df["s1_country"].unique())
    log(f"Countries present in training data: {countries_present}")
    if len(countries_present) < 2:
        log("WARNING: fewer than 2 countries present -- leave-one-country-out "
            "is not meaningful with only one country's data. Writing a report "
            "noting this rather than fabricating a cross-country split.")
        report = {
            "skipped": True,
            "reason": f"only {len(countries_present)} country present in training "
                      f"data ({countries_present}) -- cannot do leave-one-out",
        }
    else:
        results = []
        for test_country in countries_present:
            train_df = df[df["s1_country"] != test_country]
            test_df = df[df["s1_country"] == test_country]
            train_country_label = "+".join(c for c in countries_present if c != test_country)
            result = train_and_eval(train_df, test_df, feature_cols, train_country_label, test_country)
            if result:
                results.append(result)

        report = {
            "skipped": False,
            "countries_present": countries_present,
            "results": results,
            "note": "This is the ONLY honest proxy for France performance in the real "
                    "submission, since training data has zero French rows. If "
                    "cross-country macro F0.5 here is close to the same-country "
                    "GroupKFold baseline (see stage4_model_report.json), that's "
                    "evidence the model generalizes on genuine name/address "
                    "similarity signal rather than country-specific artifacts.",
        }

    report_path = os.path.join(PIPELINE_OUTPUT_DIR, "stage6_leave_one_country_out_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
