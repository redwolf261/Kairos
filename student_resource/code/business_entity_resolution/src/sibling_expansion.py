#!/usr/bin/env python3
"""
Sibling expansion -- a recall-boosting post-process step on top of a
completed flat blocking output (candidate_pairs + provenance), based on
real evidence about what separates ~89% recall (this project's measured
ceiling from 4-method token/TF-IDF/fuzzy blocking alone) from the ~99%
recall top-scoring teams are achieving on this same challenge (confirmed
via public repo descriptions found through web search: "sibling expansion
adds candidates through records that are already confidently matched").

THE IDEA: blocking sometimes finds a confident match for an S1 entity
(e.g. via exact/near-exact token overlap) but MISSES a second true match
for that same S1 because the second match's business name is a genuine
paraphrase/rebrand/heavily-abbreviated variant that no token/char-ngram
method scores well directly against the S1 query (real example found in
this project's own diagnostic: "American Oxley LLC" -> true match
"Evofaye fka American Oxley LLC", where "fka" = formerly known as -- a
LEGITIMATE company record that shares almost no useful tokens with the
S1 query text). If ANY confident candidate already exists for that S1,
however, that confident candidate's OWN text can be used as a second,
looser query -- catching sibling records that are similar to the
confident match even when they aren't similar to the original S1 text.

CONCRETE ALGORITHM (kept intentionally simple/cheap given time
constraints -- a genuinely useful first version, not the full
sophistication used by top teams' "two-stage" pipelines):
  1. For each S1 entity, find its "anchor" candidate(s): the
     highest-scoring existing candidate(s) per the already-computed
     max_block_score / num_blockers provenance (i.e. candidates multiple
     blockers agreed on, or with a very high single-method score).
  2. For each anchor, look up OTHER pool records that are highly similar
     to the ANCHOR's normalized name (not the S1's name) -- via a simple,
     fast token-overlap search against the existing name_token_index
     (reused, not rebuilt), restricted to the anchor's own country.
  3. Add any newly found records as ADDITIONAL candidates for that S1
     (tagged block_method="sibling_expansion" in provenance), subject to
     the same per-S1 candidate cap already used elsewhere in this
     pipeline.

This is a bolt-on to be run on a completed flat candidate_pairs.tsv (from
either scaled_blocking.py or test_blocking_sharded.py) -- it does NOT
require re-running the expensive blocking stage, only a cheap second pass
using structures that must be rebuilt (this script rebuilds only the
name_token_index, the cheapest of the 4 structures, per-country as needed
-- NOT the TF-IDF/fuzzy structures, to keep this fast and memory-light).

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/sibling_expansion.py \
        --candidate-pairs <path/to/candidate_pairs_flat.tsv> \
        --provenance <path/to/candidate_provenance.tsv> \
        --pool-source1 test_source1 \
        --output-dir <dir>
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_cache
import scaled_blocking as sb
from pipeline_common import log

ANCHOR_MIN_NUM_BLOCKERS = int(os.environ.get("SIBLING_ANCHOR_MIN_BLOCKERS", "2"))
SIBLING_TOP_K = int(os.environ.get("SIBLING_TOP_K", "10"))
# BUG FIX (found by direct measurement: 0 new sibling candidates on the
# first real test run despite 7,431 anchors found): this cap was set to
# 60, but the baseline blocking output being expanded already averages
# ~126 candidates/S1 -- meaning the "already at cap" break condition
# fired on essentially the FIRST anchor for nearly every S1, before any
# sibling was ever added. This cap needs to be well above whatever the
# upstream blocking stage's own per-S1 candidate count already is, since
# it caps the TOTAL (existing + sibling) count, not siblings alone. Set
# high enough to not be the binding constraint in practice -- Stage 4/5's
# scoring/threshold/one-parent-per-candidate logic is what should narrow
# candidates down, not a blunt cap at this stage.
MAX_CANDIDATES_PER_S1_AFTER_EXPANSION = int(os.environ.get("SIBLING_MAX_CANDIDATES_PER_S1", "300"))


def build_name_token_index_for_country(pool_df):
    """Cheapest of the 4 structures scaled_blocking.py builds -- reused
    logic, not reimplemented, but built ONLY for the token-index (not
    TF-IDF/fuzzy) since that's all sibling lookup needs."""
    index = defaultdict(set)
    for entity_id, tokens in zip(pool_df["entity_id"], pool_df["name_tokens"]):
        for token in set(tokens):
            if len(token) >= sb.NAME_TOKEN_MIN_LENGTH:
                index[token].add(entity_id)
    return sb._drop_oversized_buckets(index, sb.MAX_INDEX_BUCKET_SIZE, "sibling_name_token_index")


def find_similar_by_tokens(query_tokens, index, top_k):
    counts = Counter()
    for token in query_tokens:
        for entity_id in index.get(token, ()):
            counts[entity_id] += 1
    return [eid for eid, _ in counts.most_common(top_k)]


def expand_country(country, s1_country_df, pool_country_df, candidate_pairs, provenance):
    """Run sibling expansion for ONE country's worth of data -- keeps
    memory bounded the same way test_blocking_sharded.py's per-country
    discipline does."""
    log(f"  Sibling expansion for country={country}: "
        f"{len(s1_country_df):,} S1 entities, {len(pool_country_df):,} pool records")

    pool_country_df = pool_country_df.copy()
    pool_country_df["name_tokens"] = pool_country_df["business_name"].apply(sb.get_name_tokens)
    name_token_index = build_name_token_index_for_country(pool_country_df)
    pool_entity_country = dict(zip(pool_country_df["entity_id"], [country] * len(pool_country_df)))
    pool_name_lookup = dict(zip(pool_country_df["entity_id"], pool_country_df["business_name"]))
    del pool_country_df

    # anchors: candidates with strong existing provenance (multiple
    # blockers agreed, or a high single score) for S1 entities in this country
    s1_ids_this_country = set(s1_country_df["entity_id"])
    prov_this_country = provenance[provenance["s1_entity_id"].isin(s1_ids_this_country)]
    anchors = prov_this_country[prov_this_country["num_blockers"].astype(int) >= ANCHOR_MIN_NUM_BLOCKERS]
    log(f"    {len(anchors):,} anchor candidates (num_blockers >= {ANCHOR_MIN_NUM_BLOCKERS})")

    existing_by_s1 = candidate_pairs.groupby("s1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    new_rows = []
    for s1_id, anchor_cid in zip(anchors["s1_entity_id"], anchors["candidate_entity_id"]):
        anchor_name = pool_name_lookup.get(anchor_cid)
        if not anchor_name:
            continue
        anchor_tokens = sb.get_name_tokens(anchor_name)
        if not anchor_tokens:
            continue
        siblings = find_similar_by_tokens(anchor_tokens, name_token_index, SIBLING_TOP_K)
        existing = existing_by_s1.get(s1_id, set())
        for sib_id in siblings:
            if sib_id == anchor_cid or sib_id in existing:
                continue
            if len(existing) >= MAX_CANDIDATES_PER_S1_AFTER_EXPANSION:
                break
            new_rows.append({
                "s1_entity_id": s1_id, "candidate_entity_id": sib_id,
                "block_method": "sibling_expansion", "block_score": 1.0,
            })
            existing.add(sib_id)

    log(f"    {len(new_rows):,} new sibling candidates found")
    return new_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pairs", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--s1-source", required=True, help="dataset_cache file_key for S1, e.g. test_source1 or train_source1")
    parser.add_argument("--pool-source2", required=True)
    parser.add_argument("--pool-source3", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    t_start = time.time()

    log(f"Loading {args.candidate_pairs}...")
    candidate_pairs = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str, keep_default_na=False)
    log(f"  {len(candidate_pairs):,} existing candidate pairs")

    log(f"Loading {args.provenance}...")
    provenance = pd.read_csv(args.provenance, sep="\t", dtype=str, keep_default_na=False)
    log(f"  {len(provenance):,} provenance rows")

    log(f"Loading {args.s1_source}...")
    s1_all = dataset_cache.load_normalized(
        args.s1_source, columns=["entity_id", "business_name", "business_address", "country"]
    )
    countries = sorted(s1_all["country"].unique())
    log(f"Countries: {countries}")

    all_new_rows = []
    for country in countries:
        log(f"\n=== SIBLING EXPANSION SHARD: {country} ===")
        s1_country_df = s1_all[s1_all["country"] == country]

        frames = []
        for src in (args.pool_source2, args.pool_source3):
            for chunk in dataset_cache.load_normalized_lazy(
                src, columns=["entity_id", "business_name", "business_address", "country"]
            ):
                filtered = chunk[chunk["country"] == country]
                if len(filtered):
                    frames.append(filtered.copy())
        pool_country_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["entity_id", "business_name", "business_address", "country"])
        if pool_country_df.empty:
            log(f"  no pool records for {country}, skipping")
            continue

        new_rows = expand_country(country, s1_country_df, pool_country_df, candidate_pairs, provenance)
        all_new_rows.extend(new_rows)
        del pool_country_df

    log(f"\nTotal new sibling candidates across all countries: {len(all_new_rows):,}")

    new_df = pd.DataFrame(all_new_rows, columns=["s1_entity_id", "candidate_entity_id", "block_method", "block_score"])
    combined_pairs = pd.concat(
        [candidate_pairs, new_df[["s1_entity_id", "candidate_entity_id"]]], ignore_index=True
    ).drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"])

    combined_provenance_new = new_df.rename(columns={"block_method": "blocking_methods"})
    combined_provenance_new["num_blockers"] = 1
    combined_provenance_new["max_block_score"] = combined_provenance_new["block_score"]
    combined_provenance_new = combined_provenance_new[
        ["s1_entity_id", "candidate_entity_id", "blocking_methods", "num_blockers", "max_block_score"]
    ]
    combined_provenance = pd.concat([provenance, combined_provenance_new], ignore_index=True)
    combined_provenance = combined_provenance.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"], keep="first")

    os.makedirs(args.output_dir, exist_ok=True)
    out_pairs_path = os.path.join(args.output_dir, "candidate_pairs_flat_expanded.tsv")
    out_prov_path = os.path.join(args.output_dir, "candidate_provenance_expanded.tsv")
    combined_pairs.to_csv(out_pairs_path, sep="\t", index=False)
    combined_provenance.to_csv(out_prov_path, sep="\t", index=False)
    log(f"Wrote {out_pairs_path} ({len(combined_pairs):,} rows, was {len(candidate_pairs):,})")
    log(f"Wrote {out_prov_path} ({len(combined_provenance):,} rows)")
    log(f"\nTotal runtime: {round(time.time()-t_start,1)}s")


if __name__ == "__main__":
    main()
