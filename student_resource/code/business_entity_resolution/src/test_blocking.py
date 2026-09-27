#!/usr/bin/env python3
"""
FINAL SUBMISSION blocking run against the REAL TEST SET (not the training
sample scaled_blocking.py operates on). Built under time pressure near the
challenge deadline -- reuses every verified fix from scaled_blocking.py
(pool-prep worker/chunk sizing, TF-IDF vocabulary-at-scale, TF-IDF query
chunking, the Devanagari normalization fix) directly by importing that
module, rather than re-deriving any of it.

Differences from scaled_blocking.py:
  - ALL of test_source1.tsv is used (no sampling -- every test S1 entity
    needs a submission row, per the README's explicit requirement).
  - Candidate pool is test_source2.tsv + test_source3.tsv, not train.
  - No recall measurement (no ground truth exists for the test set).
  - Output written directly to pipeline_output/test_blocking/, in the
    FLAT (s1_entity_id, candidate_entity_id) shape -- build_test_pairs.py
    (or an equivalent finalize step) converts this to the required
    candidate_pairs.tsv shape afterward, same as the training pipeline.

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/test_blocking.py
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache
import scaled_blocking as sb
from pipeline_common import PIPELINE_OUTPUT_DIR, log

OUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "test_blocking")


def load_all_test_s1():
    log("Loading ALL test_source1.tsv entities (every one needs a submission row)...")
    # Only the 4 raw columns run_blocking() actually needs -- the cache's
    # default load also includes norm_name/norm_addr/name_prefix4/pin/
    # addr_last_tok (audit.py's OWN normalization pipeline, unused here
    # since scaled_blocking.py's run_blocking() recomputes its own
    # tokenization/normalization via prepare_pool_fields_parallel), which
    # cost ~0.44GB for nothing at 1.73M rows -- trimmed since every bit of
    # headroom matters given how tight memory has been on this exact pool.
    df = dataset_cache.load_normalized(
        "test_source1", columns=["entity_id", "business_name", "business_address", "country"]
    )
    log(f"  {len(df):,} test S1 entities")
    return df


def load_test_candidate_pool(countries_needed):
    """Same logic as scaled_blocking.load_candidate_pool() but against the
    TEST source files, not train."""
    log(f"Loading test_source2+3 candidate pool for countries: {sorted(countries_needed)}...")
    t0 = time.time()
    frames = []
    n_chunks = 0
    for src in ("test_source2", "test_source3"):
        for chunk in dataset_cache.load_normalized_lazy(
            src, columns=["entity_id", "business_name", "business_address", "country"]
        ):
            n_chunks += 1
            filtered = chunk[chunk["country"].isin(countries_needed)]
            if len(filtered):
                filtered = filtered.copy()
                filtered["source"] = "S2" if src == "test_source2" else "S3"
                frames.append(filtered)
            if n_chunks % 10 == 0:
                log(f"  ...scanned {n_chunks} chunks ({round(time.time()-t0,1)}s elapsed)")
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "country", "source"])
    log(f"  loaded {len(pool):,} candidate records in {round(time.time()-t0,1)}s")
    return pool


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    s1_all = load_all_test_s1()
    countries_needed = set(s1_all["country"].unique())

    pool_df = load_test_candidate_pool(countries_needed)
    candidate_pool_size = len(pool_df)
    structures = sb.build_structures(pool_df)
    del pool_df

    log(f"Running blocking for ALL {len(s1_all):,} test S1 entities...")
    candidate_pairs, provenance, per_method_counts = sb.run_blocking(s1_all, structures)

    log(f"Union candidate pairs: {len(candidate_pairs):,} "
        f"(avg {len(candidate_pairs)/len(s1_all):.1f} candidates/S1)")
    log(f"Per-method raw row counts: {per_method_counts}")

    candidate_pairs_path = os.path.join(OUT_DIR, "candidate_pairs_flat.tsv")
    provenance_path = os.path.join(OUT_DIR, "candidate_provenance.tsv")

    candidate_pairs.to_csv(candidate_pairs_path, sep="\t", index=False)
    provenance.to_csv(provenance_path, sep="\t", index=False)
    log(f"Wrote {candidate_pairs_path}")
    log(f"Wrote {provenance_path}")

    report = {
        "n_s1_total": len(s1_all),
        "countries": sorted(countries_needed),
        "candidate_pool_size": candidate_pool_size,
        "n_candidate_pairs": len(candidate_pairs),
        "avg_candidates_per_s1": round(len(candidate_pairs) / len(s1_all), 2),
        "per_method_raw_row_counts": per_method_counts,
        "total_seconds": round(time.time() - t_start, 1),
    }
    report_path = os.path.join(OUT_DIR, "test_blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log(f"\nTotal runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
