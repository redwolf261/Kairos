#!/usr/bin/env python3
"""
Closes the candidate_pairs format gap flagged in schema_contracts.py.

Blocking work in this repo (../../../../src/blocking.ipynb, teammate's code)
produces a FLAT intermediate pair table:

    s1_entity_id    candidate_entity_id
    S1-001          S2-100
    S1-001          S3-200
    S1-002          S2-300

...one row per (S1, candidate) pair -- convenient for recall measurement
against ground truth (a simple merge), which is exactly what that notebook
uses it for.

The challenge's REQUIRED candidate_pairs.tsv format is different in shape,
not just column names -- one row per SOURCE-1 ENTITY, every id that entity's
blocking surfaced joined into a single comma-separated string:

    source1_entity_id    candidate_entity_ids
    S1-001                S2-100,S3-200
    S1-002                S2-300
    S1-003                                      <- present even with 0 candidates

This module is the fuse/finalize step between those two shapes. It also
enforces the two structural rules the challenge's own validator checks
(student_resource/utils/validate_submission.py) so a file built with this
function passes validation by construction:

    1. Every required S1 entity appears in the output, even ones with zero
       candidates (empty string, not a missing row).
    2. No duplicate ids within one row's candidate list, and only S2-/S3-
       prefixed ids are ever included.

Run standalone to convert an existing flat file:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/finalize_candidate_pairs.py \
        --input <flat_pairs.tsv> \
        --test-source1 dataset/test/test_source1.tsv \
        --output output/candidate_pairs.tsv

Or import finalize_candidate_pairs() directly from pipeline code.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # reuses read_ids() for the required-S1 list


def finalize_candidate_pairs(flat_pairs, required_s1_ids, s1_col="s1_entity_id", cid_col="candidate_entity_id"):
    """Convert a flat (s1_id, candidate_id) pair table into the required
    candidate_pairs.tsv shape: one row per required S1 id, candidates
    comma-joined, S1 ids with zero candidates given an empty string.

    Parameters
    ----------
    flat_pairs : pd.DataFrame
        Columns [s1_col, cid_col], one row per (S1, candidate) pair. May
        contain duplicates and ids in any order -- both are handled.
    required_s1_ids : Iterable[str]
        Every S1 entity_id that MUST appear in the output (typically every
        id in test_source1.tsv), regardless of whether blocking found any
        candidates for it.
    s1_col, cid_col : str
        Column names in `flat_pairs` (defaults match what blocking.ipynb
        currently produces -- override if your flat table uses different
        names).

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, candidate_entity_ids],
    one row per required_s1_ids entry, in the same order as required_s1_ids.
    """
    flat_pairs = flat_pairs[[s1_col, cid_col]].copy()

    # enforce: only S2-/S3- prefixed ids are ever valid candidates (drop
    # anything else rather than let a bug silently reach the output file --
    # the challenge's validator would reject these anyway)
    bad_prefix = ~flat_pairs[cid_col].astype(str).str.startswith(("S2-", "S3-"))
    if bad_prefix.any():
        n_bad = int(bad_prefix.sum())
        print(f"WARNING: dropping {n_bad:,} candidate rows with a non-S2-/S3- "
              f"id (should not happen from a correct blocker) -- "
              f"examples: {flat_pairs.loc[bad_prefix, cid_col].unique()[:5].tolist()}")
        flat_pairs = flat_pairs[~bad_prefix]

    # de-dupe (s1, candidate) pairs before joining -- a candidate surfaced by
    # multiple blocking routes should appear once, not repeated
    flat_pairs = flat_pairs.drop_duplicates(subset=[s1_col, cid_col])

    grouped = (
        flat_pairs
        .groupby(s1_col)[cid_col]
        .apply(lambda ids: ",".join(sorted(ids)))
    )

    result = pd.DataFrame({"source1_entity_id": list(required_s1_ids)})
    result = result.merge(
        grouped.rename("candidate_entity_ids"),
        left_on="source1_entity_id", right_index=True, how="left",
    )
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

    return result


def load_required_s1_ids(test_source1_path):
    """Every S1 entity_id from test_source1.tsv, in file order -- reuses
    audit.py's existing streaming reader rather than a fresh implementation."""
    ids = []
    for chunk in audit.iter_chunks(test_source1_path, usecols=["entity_id"]):
        ids.extend(chunk["entity_id"].tolist())
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Convert a flat (s1_id, candidate_id) pair table into the "
                     "required candidate_pairs.tsv shape."
    )
    parser.add_argument("--input", required=True, help="Path to the flat intermediate pair file (.tsv or .parquet)")
    parser.add_argument("--test-source1", default="dataset/test/test_source1.tsv",
                         help="Path to test_source1.tsv (default: %(default)s)")
    parser.add_argument("--output", default="output/candidate_pairs.tsv",
                         help="Where to write the finalized candidate_pairs.tsv (default: %(default)s)")
    parser.add_argument("--s1-col", default="s1_entity_id", help="S1 id column name in --input")
    parser.add_argument("--cid-col", default="candidate_entity_id", help="candidate id column name in --input")
    args = parser.parse_args()

    print(f"Loading required S1 ids from {args.test_source1}...")
    required_ids = load_required_s1_ids(args.test_source1)
    print(f"  {len(required_ids):,} required S1 entities")

    print(f"Loading flat pair table from {args.input}...")
    if args.input.endswith(".parquet"):
        flat = pd.read_parquet(args.input)
    else:
        flat = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    print(f"  {len(flat):,} rows, columns: {list(flat.columns)}")

    result = finalize_candidate_pairs(flat, required_ids, s1_col=args.s1_col, cid_col=args.cid_col)

    n_with_candidates = (result["candidate_entity_ids"] != "").sum()
    print(f"Finalized: {len(result):,} rows "
          f"({n_with_candidates:,} with >=1 candidate, "
          f"{len(result) - n_with_candidates:,} empty)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
