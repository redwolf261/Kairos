#!/usr/bin/env python3
"""
Export a representative 10k-S1-entity experimental sample as standalone
artifacts.

IMPORTANT: this samples S1 entities UNIFORMLY AT RANDOM from the full
2,206,821-entity population -- including singletons (entities with ZERO true
matches). An earlier version of this script reused experiments.py's
sample_true_pairs(), which deliberately restricts to S1 entities that already
have >=1 true match (correct for Experiments 1/3/4, which study true-pair
similarity and need pairs to exist) -- but that means it silently drops
singletons, which are 5.585% of the real dataset and are NOT a negligible
edge case: under the challenge's F_0.5 macro-average, correctly predicting
"no match" on a singleton is worth a full 1.0, and a false merge on one
scores 0. A sample that never contains a singleton can't be used to validate
singleton-handling behavior and misrepresents the dataset's true class
balance. This version fixes that by sampling from ALL S1 entities.

Produces:

    experiment_10k/
    |-- s1_sample.tsv       -- 10,000 S1 entities, same columns as
    |                          train_source1.tsv, sampled uniformly at
    |                          random (singletons included)
    |-- true_matches.tsv    -- source1_entity_id, matched_entity_id, source
    |                          (flat form; singleton S1 entities simply have
    |                          no rows here, exactly as in the real dataset)
    |-- metadata.json       -- sample_seed, number_of_s1, singleton/S2-only/
    |                          S3-only/both counts, total_true_pairs, and a
    |                          side-by-side comparison against the full
    |                          dataset's true ratios (see
    |                          ground_truth_stats.json), so the sample's
    |                          representativeness is checked, not assumed.

Does NOT extract the ~10.3M-record S2+S3 candidate universe -- the original
files (and the shared normalized Parquet cache in dataset_cache.py) already
serve as that candidate universe for blocking experiments; duplicating it
here would be redundant and wasteful.

Run from student_resource/:
    python code/business_entity_resolution/src/export_10k_sample.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache

STUDENT_RESOURCE = audit.STUDENT_RESOURCE
OUT_DIR = os.path.join(STUDENT_RESOURCE, "experiment_10k")
N_TARGET = 10000
SEED = 42


def sample_representative_s1(n_target, seed):
    """Uniform-random sample of n_target S1 entities from the FULL
    population (singletons included), plus their true matches (empty for
    singletons) -- representative of the real dataset's class balance,
    unlike a has-match-only sample."""
    log = audit.log
    rng = np.random.RandomState(seed)

    log(f"loading S1 ({N_TARGET:,} of 2,206,821 target) + full ground truth...")
    s1_df = dataset_cache.load_normalized("train_source1")
    gt = audit.load_ground_truth_map()

    all_ids = s1_df["entity_id"].to_numpy()
    sampled_ids = set(rng.choice(all_ids, size=min(n_target, len(all_ids)), replace=False))
    log(f"sampled {len(sampled_ids):,} S1 entities uniformly at random "
        f"out of {len(all_ids):,} total (singletons included)")

    s1_sample = s1_df[s1_df["entity_id"].isin(sampled_ids)].copy()
    s1_sample = s1_sample.rename(columns={
        "entity_id": "s1_id", "business_name": "s1_business_name",
        "business_address": "s1_business_address",
    })[["s1_id", "s1_business_name", "s1_business_address", "country"]]

    pair_rows = []
    for s1_id in s1_sample["s1_id"]:
        for cid in gt.get(s1_id, ()):
            pair_rows.append((s1_id, cid, "S2" if cid.startswith("S2-") else "S3"))

    pairs = pd.DataFrame(pair_rows, columns=["s1_id", "cid", "cid_source"])
    log(f"{len(pairs):,} true pairs among the sample "
        f"({(pairs['cid_source']=='S2').sum():,} S2, {(pairs['cid_source']=='S3').sum():,} S3)")

    return s1_sample, pairs, gt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit.log(f"Exporting representative {N_TARGET:,}-entity sample (seed={SEED}) to {OUT_DIR}")

    s1_sample, pairs, gt = sample_representative_s1(N_TARGET, SEED)

    # --- s1_sample.tsv: same shape as train_source1.tsv --------------------
    s1_out = s1_sample.rename(columns={
        "s1_id": "entity_id",
        "s1_business_name": "business_name",
        "s1_business_address": "business_address",
    })[["entity_id", "business_name", "business_address", "country"]]
    s1_path = os.path.join(OUT_DIR, "s1_sample.tsv")
    s1_out.to_csv(s1_path, sep="\t", index=False)
    audit.log(f"  wrote {s1_path} ({len(s1_out):,} rows)")

    # --- true_matches.tsv: flat (source1_entity_id, matched_entity_id, source) --
    true_matches = pairs.rename(columns={
        "s1_id": "source1_entity_id", "cid": "matched_entity_id", "cid_source": "source",
    })[["source1_entity_id", "matched_entity_id", "source"]]
    tm_path = os.path.join(OUT_DIR, "true_matches.tsv")
    true_matches.to_csv(tm_path, sep="\t", index=False)
    audit.log(f"  wrote {tm_path} ({len(true_matches):,} rows)")

    # --- metadata.json, including a representativeness check ---------------
    sampled_ids = set(s1_sample["s1_id"])
    n_singleton = sum(1 for s1 in sampled_ids if not gt.get(s1))
    n_s2_only = n_s3_only = n_both = 0
    for s1 in sampled_ids:
        matches = gt.get(s1)
        if not matches:
            continue
        has_s2 = any(m.startswith("S2-") for m in matches)
        has_s3 = any(m.startswith("S3-") for m in matches)
        if has_s2 and has_s3:
            n_both += 1
        elif has_s2:
            n_s2_only += 1
        elif has_s3:
            n_s3_only += 1

    n = len(sampled_ids)
    sample_ratios = {
        "singleton_pct": round(100 * n_singleton / n, 3),
        "s2_only_pct": round(100 * n_s2_only / n, 3),
        "s3_only_pct": round(100 * n_s3_only / n, 3),
        "both_pct": round(100 * n_both / n, 3),
    }
    # ground truth's true full-dataset ratios (from audit/ground_truth_stats.json)
    full_dataset_ratios = {
        "singleton_pct": 5.585,
        "s2_only_pct": 6.481,
        "s3_only_pct": 7.454,
        "both_pct": 80.48,
    }

    metadata = {
        "sample_seed": SEED,
        "sampling_method": "uniform random over ALL S1 entities (singletons included) "
                            "-- NOT filtered to has-match-only",
        "number_of_s1": n,
        "number_of_singletons": n_singleton,
        "number_of_s2_only": n_s2_only,
        "number_of_s3_only": n_s3_only,
        "number_of_both": n_both,
        "total_true_pairs": len(pairs),
        "total_s2_matches": int((pairs["cid_source"] == "S2").sum()),
        "total_s3_matches": int((pairs["cid_source"] == "S3").sum()),
        "representativeness_check": {
            "sample_ratios": sample_ratios,
            "full_dataset_ratios": full_dataset_ratios,
            "note": "sample_ratios should be close to full_dataset_ratios; "
                    "small deviations are expected sampling noise at n=10,000 "
                    "(e.g. singleton count has a ~0.23pp standard error at "
                    "this sample size).",
        },
        "candidate_universe": {
            "note": "NOT extracted to a file -- use the full train_source2.tsv "
                    "+ train_source3.tsv (or the shared normalized Parquet "
                    "cache in audit/cache/dataset/) as the candidate universe "
                    "for blocking experiments against this sample.",
            "train_source2_total_rows": 5034616,
            "train_source3_total_rows": 5285603,
            "combined_pool_size": 5034616 + 5285603,
        },
        "source_files": {
            "s1_sample_tsv": "s1_sample.tsv",
            "true_matches_tsv": "true_matches.tsv",
        },
    }
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    audit.log(f"  wrote {meta_path}")

    audit.log("Done.")
    audit.log(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
