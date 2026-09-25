#!/usr/bin/env python3
"""
Export the 10k-S1-entity experimental sample as standalone artifacts.

Produces exactly what was asked for:

    experiment_10k/
    |-- s1_sample.tsv       -- the 10,000 (well, 10,001 -- see note below)
    |                          sampled S1 entities, same columns as
    |                          train_source1.tsv
    |-- true_matches.tsv    -- source1_entity_id, matched_entity_id, source
    |                          (one row per true S1<->S2/S3 pair; the FLAT
    |                          form, not the comma-joined ground-truth form)
    |-- metadata.json       -- sample_seed, number_of_s1, singleton/S2-only/
    |                          S3-only/both counts, total_true_pairs

Does NOT extract the ~10.3M-record S2+S3 candidate universe -- the original
files (and the shared normalized Parquet cache in dataset_cache.py) already
serve as that candidate universe for blocking experiments; duplicating it
here would be redundant and wasteful.

Note on count: sample_frac=0.0048 against 2,083,574 eligible (>=1 true
match) S1 entities rounds to 10,001, not exactly 10,000 -- close enough that
re-deriving an exact-10,000 frac isn't worth the fuss, but flagged here so
the off-by-one isn't a surprise.

Run from student_resource/:
    python code/business_entity_resolution/src/export_10k_sample.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import experiments

STUDENT_RESOURCE = audit.STUDENT_RESOURCE
OUT_DIR = os.path.join(STUDENT_RESOURCE, "experiment_10k")
SAMPLE_FRAC = 0.0048  # -> ~10,000 S1 entities (see note in docstring)
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit.log(f"Exporting 10k sample (frac={SAMPLE_FRAC}, seed={SEED}) to {OUT_DIR}")

    pairs, gt, s1_sample = experiments.sample_true_pairs(
        SAMPLE_FRAC, seed=SEED, return_context=True
    )

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

    # --- metadata.json -------------------------------------------------------
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

    metadata = {
        "sample_seed": SEED,
        "sample_frac": SAMPLE_FRAC,
        "number_of_s1": len(sampled_ids),
        "number_of_singletons": n_singleton,
        "number_of_s2_only": n_s2_only,
        "number_of_s3_only": n_s3_only,
        "number_of_both": n_both,
        "total_true_pairs": len(pairs),
        "total_s2_matches": int((pairs["cid_source"] == "S2").sum()),
        "total_s3_matches": int((pairs["cid_source"] == "S3").sum()),
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
