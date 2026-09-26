#!/usr/bin/env python3
"""
ML Challenge 2026 -- Similarity Experiments (Experiment 1 & 3)

Built on top of the shared normalized dataset cache (dataset_cache.py), which
is itself built on audit.py's file-fingerprint caching -- so re-running this
script after the first time is instant unless the source .tsvs or these
experiments' own parameters change.

Experiment 1 -- TRUE-MATCH SIMILARITY DISTRIBUTIONS
    For a sample of S1 entities with >=1 true match, fetch their actual S2/S3
    partners and compute name + address similarity features for every true
    pair. Reports the DISTRIBUTION of each feature across true matches --
    e.g. "72% of true matches have normalized names that are exactly equal";
    "88.7% have char-3-gram Jaccard >= 0.6". This tells us how much noise
    real matches actually contain, which should drive blocking/feature design
    instead of guessing.

Experiment 3 -- WHY DOES country+norm_name BLOCKING MISS ~57% OF MATCHES?
    Takes the true pairs from Experiment 1, splits them into "caught by
    country+norm_name blocking" vs "missed", and compares their similarity
    feature distributions. If the missed group clusters at low name
    similarity, that's typescript/abbreviation/reordering noise the blocking
    key can't see through. If missed pairs still have decently similar
    names, the blocking key itself (exact match) is the problem, not the
    underlying data.

Similarity features per pair (name and address, each):
    exact_raw_equal          raw string equality
    exact_norm_equal         normalized string equality
    char3_jaccard            Jaccard over character 3-grams
    char4_jaccard            Jaccard over character 4-grams
    token_jaccard            Jaccard over whitespace tokens
    edit_similarity          rapidfuzz normalized Levenshtein similarity (0-1)
    token_set_ratio          rapidfuzz token_set_ratio (0-100), robust to
                              word order / subset-of-tokens matches
    length_ratio             min(len)/max(len) of the normalized strings
    prefix4_equal            first 4 chars of normalized strings equal

Run from student_resource/:
    python code/business_entity_resolution/src/experiments.py

Env vars:
    EXPERIMENT_SAMPLE_FRAC   fraction of S1 (with >=1 true match) to sample
                             (default 0.02 -- ~35k S1 entities, ~120k pairs)
    NO_CACHE=1               bypass cache, recompute from scratch

Outputs land in `audit/`:
    experiment1_true_match_similarity.json
    experiment3_missed_by_blocking.json
    experiments_summary.txt
"""

import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache

AUDIT_DIR = audit.AUDIT_DIR
EXPERIMENT_SAMPLE_FRAC = float(os.environ.get("EXPERIMENT_SAMPLE_FRAC", "0.02"))


# ---------------------------------------------------------------------------
# Similarity features
# ---------------------------------------------------------------------------

def char_ngrams(s, n):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def pair_similarity_features(s1_val, s2_val, prefix):
    """Compute the standard feature set for one (s1_val, s2_val) string pair
    (already normalized), keyed with the given prefix ('name_' or 'addr_')."""
    s1_val = s1_val or ""
    s2_val = s2_val or ""

    tok1 = set(s1_val.split())
    tok2 = set(s2_val.split())
    tri1 = char_ngrams(s1_val, 3)
    tri2 = char_ngrams(s2_val, 3)
    quad1 = char_ngrams(s1_val, 4)
    quad2 = char_ngrams(s2_val, 4)

    max_len = max(len(s1_val), len(s2_val))
    min_len = min(len(s1_val), len(s2_val))

    return {
        f"{prefix}exact_norm_equal": s1_val == s2_val,
        f"{prefix}char3_jaccard": jaccard(tri1, tri2),
        f"{prefix}char4_jaccard": jaccard(quad1, quad2),
        f"{prefix}token_jaccard": jaccard(tok1, tok2),
        f"{prefix}edit_similarity": Levenshtein.normalized_similarity(s1_val, s2_val) if (s1_val or s2_val) else 1.0,
        f"{prefix}token_set_ratio": fuzz.token_set_ratio(s1_val, s2_val) / 100.0,
        f"{prefix}length_ratio": (min_len / max_len) if max_len else 1.0,
        f"{prefix}prefix4_equal": s1_val[:4] == s2_val[:4] if (s1_val and s2_val) else False,
    }


def compute_pair_features(row):
    """row has s1_norm_name, s2_norm_name, s1_norm_addr, s2_norm_addr,
    s1_business_name, s2_business_name (raw, for exact_raw_equal)."""
    feats = {}
    feats["name_exact_raw_equal"] = row["s1_business_name"] == row["s2_business_name"]
    feats.update(pair_similarity_features(row["s1_norm_name"], row["s2_norm_name"], "name_"))
    feats["addr_exact_raw_equal"] = row["s1_business_address"] == row["s2_business_address"]
    feats.update(pair_similarity_features(row["s1_norm_addr"], row["s2_norm_addr"], "addr_"))
    return feats


# ---------------------------------------------------------------------------
# Sampling: pick S1 entities with >=1 true match, build the flat pair table
# with both sides' normalized fields already joined in from the shared cache.
# ---------------------------------------------------------------------------

def sample_true_pairs(sample_frac, seed=42, return_context=False):
    """Returns a DataFrame with one row per (S1, matched S2-or-S3) true pair,
    columns: s1_id, cid, cid_source ('S2'/'S3'), s1_business_name,
    s2_business_name, s1_norm_name, s2_norm_name, s1_business_address,
    s2_business_address, s1_norm_addr, s2_norm_addr, country.

    With return_context=True, also returns (gt, s1_sample) so callers (e.g.
    Experiment 2's false-pair generation) can reuse the same S1 sample and
    ground-truth map without re-sampling or re-scanning ground truth."""
    rng = np.random.RandomState(seed)

    log = audit.log
    log(f"[experiments] loading S1 + ground truth (sample_frac={sample_frac})...")
    s1_df = dataset_cache.load_normalized("train_source1")
    # NOTE: only_ids is intentionally omitted here -- passing the full S1 id
    # set (all 2.2M ids) as an "only_ids" filter is not a real filter (every
    # ground-truth row's S1 id is by definition in that set), it just forces
    # a slow set-membership .isin() check against ~2.2M ground-truth rows for
    # no benefit. Loading the full map directly is faster.
    gt = audit.load_ground_truth_map()

    has_match_ids = np.array([s1 for s1 in s1_df["entity_id"] if gt.get(s1)])
    n_sample = max(1, int(len(has_match_ids) * sample_frac))
    sampled_ids = set(rng.choice(has_match_ids, size=min(n_sample, len(has_match_ids)), replace=False))
    log(f"[experiments] sampled {len(sampled_ids):,} S1 entities (with >=1 true match) "
        f"out of {len(has_match_ids):,} eligible")

    s1_sample = s1_df[s1_df["entity_id"].isin(sampled_ids)].copy()
    s1_sample = s1_sample.rename(columns={
        "entity_id": "s1_id", "business_name": "s1_business_name",
        "business_address": "s1_business_address", "norm_name": "s1_norm_name",
        "norm_addr": "s1_norm_addr",
    })[["s1_id", "s1_business_name", "s1_business_address", "s1_norm_name", "s1_norm_addr", "country"]]

    # flat (s1_id, cid) true-pair table
    pair_rows = []
    for s1_id in s1_sample["s1_id"]:
        for cid in gt.get(s1_id, ()):
            pair_rows.append((s1_id, cid, "S2" if cid.startswith("S2-") else "S3"))
    pairs = pd.DataFrame(pair_rows, columns=["s1_id", "cid", "cid_source"])
    log(f"[experiments] {len(pairs):,} true pairs to evaluate "
        f"({(pairs['cid_source']=='S2').sum():,} S2, {(pairs['cid_source']=='S3').sum():,} S3)")

    # fetch the matched S2/S3 records via the shared cache's filtered load
    # (pushed-down entity_id filter -- never loads the full 5M-row files)
    s2_ids = set(pairs.loc[pairs["cid_source"] == "S2", "cid"])
    s3_ids = set(pairs.loc[pairs["cid_source"] == "S3", "cid"])
    log(f"[experiments] fetching {len(s2_ids):,} S2 + {len(s3_ids):,} S3 matched records "
        f"from cache (filtered load)...")

    t0 = time.time()
    s2_df = dataset_cache.load_normalized("train_source2", entity_ids=s2_ids) if s2_ids else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "norm_name", "norm_addr"])
    s3_df = dataset_cache.load_normalized("train_source3", entity_ids=s3_ids) if s3_ids else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "norm_name", "norm_addr"])
    log(f"[experiments] fetched matched records in {round(time.time()-t0,1)}s "
        f"(S2: {len(s2_df):,} rows, S3: {len(s3_df):,} rows)")

    cid_df = pd.concat([s2_df, s3_df], ignore_index=True)
    cid_df = cid_df.rename(columns={
        "entity_id": "cid", "business_name": "s2_business_name",
        "business_address": "s2_business_address", "norm_name": "s2_norm_name",
        "norm_addr": "s2_norm_addr",
    })[["cid", "s2_business_name", "s2_business_address", "s2_norm_name", "s2_norm_addr"]]

    pairs = pairs.merge(s1_sample, on="s1_id", how="left")
    pairs = pairs.merge(cid_df, on="cid", how="left")
    # drop any pair whose candidate record wasn't found (shouldn't happen for
    # ground-truth pairs, but guards against a data inconsistency)
    missing = pairs["s2_business_name"].isna()
    if missing.any():
        log(f"[experiments] WARNING: {missing.sum():,} true pairs reference a "
            f"candidate id not found in its source file -- dropping them")
        pairs = pairs[~missing].reset_index(drop=True)

    if return_context:
        return pairs, gt, s1_sample
    return pairs


# ---------------------------------------------------------------------------
# Feature computation -- parallelized across chunks of the pair table
# ---------------------------------------------------------------------------

def _compute_features_chunk(rows_dict_list):
    """Worker: compute similarity features for a list of pair-record dicts.
    Runs in a separate process; takes/returns only plain picklable types."""
    return [compute_pair_features(row) for row in rows_dict_list]


def compute_features_parallel(pairs_df):
    """IMPORTANT: chunks are dispatched to worker processes and MUST be
    reassembled in the same order they were submitted, not the order they
    finish in. Worker completion order is inherently non-deterministic
    (depends on OS scheduling, chunk size variance, etc.), so iterating
    futures via as_completed() and extending a flat list -- as an earlier
    version of this function did -- silently pairs each row's original
    data with a DIFFERENT row's computed features (chunk k's features land
    at whatever position chunk k happened to finish in, not position k).
    That bug was caught because it made two runs on the IDENTICAL input
    produce different feature values, which should be impossible for a
    pure function -- a strong signal to look for exactly this class of bug.
    Fix: keep futures indexed by their submission order and reassemble in
    that order, regardless of completion order."""
    log = audit.log
    n = len(pairs_df)
    log(f"[experiments] computing similarity features for {n:,} pairs "
        f"(parallel across {audit.MAX_WORKERS} workers)...")
    t0 = time.time()

    records = pairs_df.to_dict("records")
    n_workers = audit.MAX_WORKERS
    chunk_size = max(1, -(-len(records) // n_workers))  # ceil division
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    results_by_chunk_index = {}
    with audit.make_executor() as ex:
        future_to_index = {ex.submit(_compute_features_chunk, c): i for i, c in enumerate(chunks)}
        for fut in audit.as_completed(future_to_index):
            results_by_chunk_index[future_to_index[fut]] = fut.result()

    all_features = []
    for i in range(len(chunks)):
        all_features.extend(results_by_chunk_index[i])

    feat_df = pd.DataFrame(all_features)
    log(f"[experiments] features computed in {round(time.time()-t0,1)}s")
    return pd.concat([pairs_df.reset_index(drop=True), feat_df], axis=1)


# ---------------------------------------------------------------------------
# Experiment 1: true-match similarity distributions
# ---------------------------------------------------------------------------

FEATURE_THRESHOLDS = {
    "name_char3_jaccard": [0.9, 0.8, 0.6, 0.4, 0.2],
    "name_char4_jaccard": [0.9, 0.8, 0.6, 0.4, 0.2],
    "name_token_jaccard": [0.9, 0.7, 0.5, 0.3],
    "name_edit_similarity": [0.9, 0.8, 0.6, 0.4],
    "name_token_set_ratio": [0.9, 0.8, 0.6, 0.4],
    "name_length_ratio": [0.9, 0.7, 0.5],
    "addr_char3_jaccard": [0.8, 0.6, 0.4, 0.2],
    "addr_token_jaccard": [0.7, 0.5, 0.3],
    "addr_edit_similarity": [0.8, 0.6, 0.4],
}


def summarize_distributions(df, label):
    out = {"n_pairs": len(df), "label": label}
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in bool_cols:
        out[f"{c}_pct"] = round(100 * float(df[c].mean()), 2)

    for feat, thresholds in FEATURE_THRESHOLDS.items():
        if feat not in df.columns:
            continue
        out[feat] = {
            "mean": round(float(df[feat].mean()), 4),
            "median": round(float(df[feat].median()), 4),
            "p10": round(float(df[feat].quantile(0.10)), 4),
            "thresholds_pct_ge": {
                f">={t}": round(100 * float((df[feat] >= t).mean()), 2) for t in thresholds
            },
        }

    # "strong signal" composite: name OR address similarity is high
    if "name_char3_jaccard" in df.columns and "addr_char3_jaccard" in df.columns:
        strong = (df["name_char3_jaccard"] >= 0.6) | (df["addr_char3_jaccard"] >= 0.6)
        out["name_or_addr_char3_jaccard_ge_0.6_pct"] = round(100 * float(strong.mean()), 2)

    return out


def run_experiment1(pairs_with_features):
    audit.log("[Experiment 1] summarizing true-match similarity distributions...")
    return summarize_distributions(pairs_with_features, "true_matches")


# ---------------------------------------------------------------------------
# Experiment 3: why does country+norm_name blocking miss matches?
# ---------------------------------------------------------------------------

def run_experiment3(pairs_with_features):
    audit.log("[Experiment 3] splitting true matches by country+norm_name blocking hit/miss...")
    df = pairs_with_features
    # The blocking strategy under test is "country + normalized name". We
    # only carried S1's country into this pair table (the candidate's own
    # country wasn't fetched), but for TRUE matches the country component is
    # satisfied in the overwhelming majority of cases (matches are almost
    # always same-country by construction -- see audit's blocking_stats.json
    # "country" strategy showing ~100% recall), so "caught" reduces to the
    # name-equality condition alone.
    caught_mask = df["s1_norm_name"] == df["s2_norm_name"]

    caught_df = df[caught_mask]
    missed_df = df[~caught_mask]

    result = {
        "caught_pct": round(100 * float(caught_mask.mean()), 2),
        "missed_pct": round(100 * float((~caught_mask).mean()), 2),
        "caught": summarize_distributions(caught_df, "caught_by_blocking"),
        "missed": summarize_distributions(missed_df, "missed_by_blocking"),
    }

    # categorize missed pairs by likely noise type using cheap heuristics
    if len(missed_df):
        m = missed_df
        categories = Counter()
        for _, row in m.iterrows():
            n1, n2 = row["s1_norm_name"], row["s2_norm_name"]
            t1, t2 = set(n1.split()), set(n2.split())
            if not n1 or not n2:
                categories["name_missing"] += 1
            elif t1 == t2:
                categories["word_order_only"] += 1
            elif row["name_token_jaccard"] >= 0.8:
                categories["minor_token_diff_likely_suffix_or_typo"] += 1
            elif row["name_char3_jaccard"] >= 0.5:
                categories["moderate_char_overlap_likely_abbrev_or_typo"] += 1
            elif row["name_char3_jaccard"] >= 0.2:
                categories["low_char_overlap"] += 1
            else:
                categories["name_completely_different_or_script_mismatch"] += 1
        result["missed_categorization"] = {
            k: {"count": v, "pct_of_missed": round(100 * v / len(m), 2)}
            for k, v in categories.most_common()
        }

    return result


# ---------------------------------------------------------------------------
# Experiment 2: true-pair vs false-pair similarity distributions
# ---------------------------------------------------------------------------
#
# "False pairs" here means: candidates a cheap, realistic blocking strategy
# WOULD surface for an S1 entity, that are NOT its true match. Random pairs
# from the whole dataset would be trivially dissimilar and tell us nothing
# useful -- the pairs a classifier will actually have to discriminate are the
# ones that already passed blocking, i.e. share country + a loose name key.
# We use country + name_prefix4 (the strategy Experiment/audit step 6 already
# showed gives ~79% recall at ~7k candidates/S1) as the source of plausible
# negatives.

def sample_false_pairs(s1_sample, gt, max_negatives_per_s1=5, seed=43):
    """For each S1 in s1_sample, find candidates sharing country+name_prefix4
    that are NOT true matches, sample up to max_negatives_per_s1 of them, and
    return a DataFrame in the same shape as sample_true_pairs' output."""
    log = audit.log
    rng = np.random.RandomState(seed)

    log(f"[experiments] Experiment 2: building country+name_prefix4 candidate "
        f"pools for false-pair sampling...")
    t0 = time.time()

    s1_kf = s1_sample[["s1_id", "country", "s1_norm_name"]].copy()
    s1_kf["name_prefix4"] = s1_kf["s1_norm_name"].str.replace(" ", "", regex=False).str.slice(0, 4)
    s1_kf["block_key"] = s1_kf["country"] + "||" + s1_kf["name_prefix4"]
    s1_kf = s1_kf[s1_kf["name_prefix4"] != ""]

    # stream S2+S3 (via the shared cache's chunked lazy loader -- no full
    # in-memory load of either 5M-row file) and collect, per block_key, a
    # bounded reservoir of candidate ids (bounded so a very common prefix
    # like "US||phys" doesn't blow memory the way the raw audit script did
    # before its fix).
    MAX_POOL_PER_KEY = 50
    candidate_pool = defaultdict(list)  # block_key -> [cid, ...] (bounded)
    for src in ("train_source2", "train_source3"):
        for chunk in dataset_cache.load_normalized_lazy(
            src, columns=["entity_id", "country", "norm_name"]
        ):
            name_prefix4 = chunk["norm_name"].str.replace(" ", "", regex=False).str.slice(0, 4)
            block_key = chunk["country"] + "||" + name_prefix4
            for k, cid in zip(block_key, chunk["entity_id"]):
                bucket = candidate_pool[k]
                if len(bucket) < MAX_POOL_PER_KEY:
                    bucket.append(cid)
    log(f"[experiments] candidate pools built in {round(time.time()-t0,1)}s "
        f"({len(candidate_pool):,} distinct block keys)")

    # for each S1, sample up to max_negatives_per_s1 candidates from its pool
    # that are NOT a true match
    neg_rows = []
    for s1_id, key in zip(s1_kf["s1_id"], s1_kf["block_key"]):
        pool = candidate_pool.get(key)
        if not pool:
            continue
        true_matches = gt.get(s1_id, frozenset())
        candidates = [c for c in pool if c not in true_matches]
        if not candidates:
            continue
        n_take = min(max_negatives_per_s1, len(candidates))
        chosen = rng.choice(candidates, size=n_take, replace=False)
        for cid in chosen:
            neg_rows.append((s1_id, cid, "S2" if cid.startswith("S2-") else "S3"))

    neg_pairs = pd.DataFrame(neg_rows, columns=["s1_id", "cid", "cid_source"])
    log(f"[experiments] sampled {len(neg_pairs):,} false pairs "
        f"(blocking candidates that are NOT true matches)")

    # fetch the candidate records the same way sample_true_pairs does
    s2_ids = set(neg_pairs.loc[neg_pairs["cid_source"] == "S2", "cid"])
    s3_ids = set(neg_pairs.loc[neg_pairs["cid_source"] == "S3", "cid"])
    s2_df = dataset_cache.load_normalized("train_source2", entity_ids=s2_ids) if s2_ids else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "norm_name", "norm_addr"])
    s3_df = dataset_cache.load_normalized("train_source3", entity_ids=s3_ids) if s3_ids else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "norm_name", "norm_addr"])
    cid_df = pd.concat([s2_df, s3_df], ignore_index=True).rename(columns={
        "entity_id": "cid", "business_name": "s2_business_name",
        "business_address": "s2_business_address", "norm_name": "s2_norm_name",
        "norm_addr": "s2_norm_addr",
    })[["cid", "s2_business_name", "s2_business_address", "s2_norm_name", "s2_norm_addr"]]

    s1_side = s1_sample[["s1_id", "s1_business_name", "s1_business_address",
                          "s1_norm_name", "s1_norm_addr", "country"]]
    neg_pairs = neg_pairs.merge(s1_side, on="s1_id", how="left")
    neg_pairs = neg_pairs.merge(cid_df, on="cid", how="left")
    neg_pairs = neg_pairs[neg_pairs["s2_business_name"].notna()].reset_index(drop=True)
    return neg_pairs


def run_experiment2(true_features, false_features):
    audit.log("[Experiment 2] comparing true-pair vs false-pair (blocking candidate) distributions...")
    true_summary = summarize_distributions(true_features, "true_pairs")
    false_summary = summarize_distributions(false_features, "false_pairs_from_blocking")

    # explicit side-by-side histogram for the headline feature, binned --
    # this is the "do the distributions separate cleanly" check
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    histogram = {}
    for feat in ("name_char3_jaccard", "addr_char3_jaccard", "name_token_set_ratio"):
        t_counts = pd.cut(true_features[feat], bins=bins, labels=bin_labels, include_lowest=True).value_counts()
        f_counts = pd.cut(false_features[feat], bins=bins, labels=bin_labels, include_lowest=True).value_counts()
        histogram[feat] = {
            label: {
                "true_pct": round(100 * float(t_counts.get(label, 0)) / len(true_features), 2) if len(true_features) else None,
                "false_pct": round(100 * float(f_counts.get(label, 0)) / len(false_features), 2) if len(false_features) else None,
            }
            for label in bin_labels
        }

    return {
        "true_pairs": true_summary,
        "false_pairs": false_summary,
        "histogram_true_vs_false": histogram,
    }


# ---------------------------------------------------------------------------
# Experiment 4: transliteration / script analysis
# ---------------------------------------------------------------------------
#
# Detects script mixture using Unicode codepoint ranges (pure computation on
# the strings we already have -- NOT an external lookup/API, which the
# challenge rules prohibit). Devanagari block: U+0900-U+097F. Extend
# SCRIPT_RANGES if other non-Latin scripts turn out to matter.

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "latin_basic": (0x0041, 0x007A),  # rough; NFKC-folded lowercase covers most
}


def classify_script(s):
    if not s:
        return "empty"
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in s)
    has_latin = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "other"


def run_experiment4(pairs_with_features):
    audit.log("[Experiment 4] transliteration / script analysis on India true matches...")
    india = pairs_with_features[pairs_with_features["country"] == "India"].copy()
    unequal = india[india["s1_norm_name"] != india["s2_norm_name"]].copy()

    unequal["s1_script"] = unequal["s1_business_name"].map(classify_script)
    unequal["s2_script"] = unequal["s2_business_name"].map(classify_script)
    unequal["script_pair"] = unequal["s1_script"] + "_vs_" + unequal["s2_script"]

    script_pair_counts = unequal["script_pair"].value_counts().to_dict()
    cross_script = unequal[
        ((unequal["s1_script"] == "devanagari") & (unequal["s2_script"] == "latin")) |
        ((unequal["s1_script"] == "latin") & (unequal["s2_script"] == "devanagari"))
    ]

    result = {
        "india_true_matches": int(len(india)),
        "india_true_matches_with_unequal_norm_names": int(len(unequal)),
        "unequal_pct_of_india": round(100 * len(unequal) / len(india), 2) if len(india) else None,
        "script_pair_distribution": {k: int(v) for k, v in script_pair_counts.items()},
        "cross_script_latin_devanagari_count": int(len(cross_script)),
        "cross_script_pct_of_unequal": round(100 * len(cross_script) / len(unequal), 2) if len(unequal) else None,
        "examples_cross_script": [],
    }

    if len(cross_script):
        sample_n = min(15, len(cross_script))
        examples = cross_script.sample(sample_n, random_state=1)
        result["examples_cross_script"] = [
            {"s1_name": r["s1_business_name"], "s2_name": r["s2_business_name"],
             "s1_script": r["s1_script"], "s2_script": r["s2_script"]}
            for _, r in examples.iterrows()
        ]

    return result


# ---------------------------------------------------------------------------
# Experiment 5: systematic (union) blocking recall sweep
# ---------------------------------------------------------------------------
#
# Extends audit.py's single-strategy blocking_experiments with UNIONS of
# strategies, measured incrementally (B1, B1+B2, B1+B2+B3, ...) so we can see
# the marginal recall each added blocking key buys, and at what candidate-
# count cost. Reuses the true-pair sample already computed for Experiment 1/3
# instead of re-sampling.

def _rare_token(norm_name, token_freq):
    tokens = [t for t in norm_name.split() if len(t) > 1]
    if not tokens:
        return ""
    return min(tokens, key=lambda t: token_freq.get(t, 0))


def run_experiment5(s1_sample, gt, sample_frac_note):
    audit.log("[Experiment 5] union blocking recall sweep...")
    t0 = time.time()

    s1 = s1_sample.copy()
    s1["name_prefix3"] = s1["s1_norm_name"].str.replace(" ", "", regex=False).str.slice(0, 3)
    s1["name_prefix4"] = s1["s1_norm_name"].str.replace(" ", "", regex=False).str.slice(0, 4)

    # rare-token key: needs a global token frequency table -- build from a
    # sample of the pool via the shared cache's lazy loader (bounded rows)
    audit.log("    building token frequency table for rare-token blocking key...")
    token_freq = Counter()
    rows_seen = 0
    for src in ("train_source2", "train_source3"):
        for chunk in dataset_cache.load_normalized_lazy(src, columns=["norm_name"]):
            for s in chunk["norm_name"]:
                if s:
                    token_freq.update(set(s.split()))
            rows_seen += len(chunk)
            if rows_seen >= 1_000_000:
                break
        if rows_seen >= 1_000_000:
            break
    s1["rare_token"] = s1["s1_norm_name"].map(lambda s: _rare_token(s, token_freq))

    # combined key is blanked out (-> "") when the second component is empty,
    # so the later `df["key"] != ""` filter correctly excludes S1 rows that
    # have no usable value for that particular blocking feature (e.g. no
    # extractable rare token) instead of matching them on "country||" alone.
    strategy_components = {
        "B1_country_norm_name": s1["s1_norm_name"],
        "B2_country_prefix4": s1["name_prefix4"],
        "B3_country_prefix3": s1["name_prefix3"],
        "B4_country_rare_token": s1["rare_token"],
    }
    for name, component in strategy_components.items():
        combined = s1["country"] + "||" + component
        s1[name] = combined.where(component != "", "")

    # build S1-side key frames per strategy for merge-based candidate lookup,
    # same pattern as audit.py's blocking_experiments but now tracking UNION
    # membership across strategies incrementally.
    strat_names = list(strategy_components.keys())
    s1_key_frames = {}
    for name in strat_names:
        df = pd.DataFrame({"key": s1[name].to_numpy(), "s1_id": s1["s1_id"].to_numpy()})
        df = df[df["key"] != ""]
        s1_key_frames[name] = df

    # candidate ids per (strategy, s1_id), accumulated across the pool scan
    candidate_ids = {name: defaultdict(set) for name in strat_names}
    total_pool_size = 0
    for src in ("train_source2", "train_source3"):
        for chunk in dataset_cache.load_normalized_lazy(
            src, columns=["entity_id", "country", "norm_name", "name_prefix4"]
        ):
            total_pool_size += len(chunk)
            name_prefix3 = chunk["norm_name"].str.replace(" ", "", regex=False).str.slice(0, 3)
            pool_keys = {
                "B1_country_norm_name": chunk["country"] + "||" + chunk["norm_name"],
                "B2_country_prefix4": chunk["country"] + "||" + chunk["name_prefix4"],
                "B3_country_prefix3": chunk["country"] + "||" + name_prefix3,
                "B4_country_rare_token": chunk["country"] + "||" + chunk["norm_name"].map(
                    lambda s: _rare_token(s, token_freq)),
            }
            for strat_name, keys in pool_keys.items():
                s1_kf = s1_key_frames[strat_name]
                if s1_kf.empty:
                    continue
                pool_kf = pd.DataFrame({"key": keys.to_numpy(), "cid": chunk["entity_id"].to_numpy()})
                merged = pool_kf.merge(s1_kf, on="key", how="inner")
                if merged.empty:
                    continue
                cd = candidate_ids[strat_name]
                for s1_id, sub in merged.groupby("s1_id", sort=False)["cid"]:
                    cd[s1_id].update(sub.tolist())

    audit.log(f"    pool scan done in {round(time.time()-t0,1)}s, pool_size={total_pool_size:,}")

    # Incremental union B1, B1+B2, B1+B2+B3, B1+B2+B3+B4 -- computed with ONE
    # union per S1 PER STAGE (not re-unioning/re-scanning previous stages'
    # already-large sets on every subsequent stage). The earlier version kept
    # a single running_ids[s1_id] set and re-grew/re-diffed it 4 times, which
    # meant B3/B4's large, unselective candidate pools got unioned into an
    # already-large set repeatedly; each `cids - running_ids[s1_id]` diff
    # itself costs O(len(cids)), so the total work grows with the SUM of all
    # stages' pool sizes, not just the current stage's -- that's what made
    # this take 20+ minutes. Fix: keep each stage's candidate sets separate,
    # and at each reporting point build ONE fresh union of "all strategies up
    # to and including this stage" directly from those separate per-stage
    # dicts (still O(current cumulative size) per S1, but computed once, not
    # accumulated across repeated full-set unions).
    order = strat_names
    only_eligible = [s1_id for s1_id in s1["s1_id"] if gt.get(s1_id)]
    total_true_all = sum(len(gt[s1_id]) for s1_id in only_eligible)

    results = {}
    for i, strat_name in enumerate(order):
        active_strats = order[: i + 1]
        hits = 0
        cand_counts = []
        for s1_id in only_eligible:
            true_matches = gt[s1_id]
            # union only the strategies active at this stage, for this one S1
            union_here = set()
            for s in active_strats:
                s_cids = candidate_ids[s].get(s1_id)
                if s_cids:
                    union_here |= s_cids
            cand_counts.append(len(union_here))
            hits += len(true_matches & union_here)

        recall = hits / total_true_all if total_true_all else None
        avg_c = float(np.mean(cand_counts)) if cand_counts else 0.0
        label = " + ".join(active_strats)
        results[label] = {
            "recall_pct": round(100 * recall, 3) if recall is not None else None,
            "avg_candidates_per_s1": round(avg_c, 1),
            "median_candidates_per_s1": float(np.median(cand_counts)) if cand_counts else 0,
            "p95_candidates_per_s1": float(np.percentile(cand_counts, 95)) if cand_counts else 0,
            "p99_candidates_per_s1": float(np.percentile(cand_counts, 99)) if cand_counts else 0,
            "max_candidates_per_s1": int(np.max(cand_counts)) if cand_counts else 0,
            "reduction_ratio_pct": round(100 * (1 - avg_c / total_pool_size), 4) if total_pool_size else None,
        }
        audit.log(f"    union[{label}]: recall={results[label]['recall_pct']}%, "
                  f"avg_candidates={results[label]['avg_candidates_per_s1']}")

    results["_meta"] = {
        "s1_sample_size": len(s1),
        "candidate_pool_size": total_pool_size,
        "sample_frac": sample_frac_note,
        "seconds": round(time.time() - t0, 1),
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXP5_SAMPLE_FRAC = float(os.environ.get("EXP5_SAMPLE_FRAC", "0.01"))
EXP2_MAX_NEG_PER_S1 = int(os.environ.get("EXP2_MAX_NEG_PER_S1", "5"))


def main():
    import json
    t_start = time.time()
    audit.log(f"Experiments 1-5 -- sample_frac={EXPERIMENT_SAMPLE_FRAC}, "
              f"exp5_sample_frac={EXP5_SAMPLE_FRAC}")

    cache_key_files = ["train_source1", "train_source2", "train_source3", "train_ground_truth"]

    # Experiments 1, 3, 4 share the same true-pair sample + features
    cached_134 = audit.load_cached("experiments_1_3_4", cache_key_files, sample_frac=EXPERIMENT_SAMPLE_FRAC)
    if cached_134 is not None:
        exp1, exp3, exp4 = cached_134["experiment1"], cached_134["experiment3"], cached_134["experiment4"]
        # Experiment 2 needs the raw sample context (s1_sample, gt) even on a
        # cache hit for 1/3/4, since it's cached separately -- rebuild context
        # cheaply (fast now that ground-truth loading isn't over-filtered).
        _, gt, s1_sample = sample_true_pairs(EXPERIMENT_SAMPLE_FRAC, return_context=True)
        pairs_with_features = None  # not needed again on a full cache hit
    else:
        pairs, gt, s1_sample = sample_true_pairs(EXPERIMENT_SAMPLE_FRAC, return_context=True)
        pairs_with_features = compute_features_parallel(pairs)
        exp1 = run_experiment1(pairs_with_features)
        exp3 = run_experiment3(pairs_with_features)
        exp4 = run_experiment4(pairs_with_features)
        audit.save_cache("experiments_1_3_4", cache_key_files, {
            "experiment1": exp1, "experiment3": exp3, "experiment4": exp4,
        }, sample_frac=EXPERIMENT_SAMPLE_FRAC)

    # Experiment 2: true vs false pairs (false pairs from realistic blocking)
    cached_2 = audit.load_cached("experiment2", cache_key_files,
                                  sample_frac=EXPERIMENT_SAMPLE_FRAC, max_neg=EXP2_MAX_NEG_PER_S1)
    if cached_2 is not None:
        exp2 = cached_2
    else:
        if pairs_with_features is None:
            pairs = sample_true_pairs(EXPERIMENT_SAMPLE_FRAC)
            pairs_with_features = compute_features_parallel(pairs)
        false_pairs = sample_false_pairs(s1_sample, gt, max_negatives_per_s1=EXP2_MAX_NEG_PER_S1)
        false_features = compute_features_parallel(false_pairs)
        exp2 = run_experiment2(pairs_with_features, false_features)
        audit.save_cache("experiment2", cache_key_files, exp2,
                          sample_frac=EXPERIMENT_SAMPLE_FRAC, max_neg=EXP2_MAX_NEG_PER_S1)

    # Experiment 5: union blocking sweep (its own, typically smaller, sample)
    cached_5 = audit.load_cached("experiment5", cache_key_files, sample_frac=EXP5_SAMPLE_FRAC)
    if cached_5 is not None:
        exp5 = cached_5
    else:
        if abs(EXP5_SAMPLE_FRAC - EXPERIMENT_SAMPLE_FRAC) < 1e-12:
            exp5_s1_sample, exp5_gt = s1_sample, gt
        else:
            _, exp5_gt, exp5_s1_sample = sample_true_pairs(EXP5_SAMPLE_FRAC, return_context=True)
        exp5 = run_experiment5(exp5_s1_sample, exp5_gt, EXP5_SAMPLE_FRAC)
        audit.save_cache("experiment5", cache_key_files, exp5, sample_frac=EXP5_SAMPLE_FRAC)

    outputs = {
        "experiment1_true_match_similarity.json": exp1,
        "experiment2_true_vs_false.json": exp2,
        "experiment3_missed_by_blocking.json": exp3,
        "experiment4_transliteration.json": exp4,
        "experiment5_union_blocking_sweep.json": exp5,
    }
    for filename, obj in outputs.items():
        with open(os.path.join(AUDIT_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

    _write_text_summary(exp1, exp2, exp3, exp4, exp5)
    audit.log(f"Done. Total runtime: {round(time.time()-t_start,1)}s")


def _summary_block(d, indent="  "):
    lines = []
    for k in sorted(d):
        if k in ("n_pairs", "label"):
            continue
        v = d[k]
        if isinstance(v, dict) and "thresholds_pct_ge" in v:
            lines.append(f"{indent}{k}: mean={v['mean']} median={v['median']}")
            for tk, tv in v["thresholds_pct_ge"].items():
                lines.append(f"{indent}    {tk}: {tv}%")
        elif isinstance(v, (int, float, str, bool)):
            lines.append(f"{indent}{k}: {v}")
    return lines


def _write_text_summary(exp1, exp2, exp3, exp4, exp5):
    lines = []
    lines.append("=" * 78)
    lines.append("EXPERIMENT 1 -- TRUE-MATCH SIMILARITY DISTRIBUTIONS")
    lines.append("=" * 78)
    lines.append(f"n_pairs: {exp1['n_pairs']:,}")
    lines.extend(_summary_block(exp1))
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 2 -- TRUE PAIRS vs FALSE PAIRS (BLOCKING CANDIDATES)")
    lines.append("=" * 78)
    lines.append(f"true_pairs n={exp2['true_pairs']['n_pairs']:,}   "
                 f"false_pairs n={exp2['false_pairs']['n_pairs']:,}")
    lines.append("Histogram (TRUE% / FALSE% per bin):")
    for feat, bins in exp2["histogram_true_vs_false"].items():
        lines.append(f"  {feat}:")
        for bin_label, vals in bins.items():
            lines.append(f"      {bin_label}: true={vals['true_pct']}%  false={vals['false_pct']}%")
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 3 -- WHY country+norm_name BLOCKING MISSES MATCHES")
    lines.append("=" * 78)
    lines.append(f"caught_pct: {exp3['caught_pct']}%   missed_pct: {exp3['missed_pct']}%")
    if "missed_categorization" in exp3:
        lines.append("Missed-match categorization:")
        for cat, info in exp3["missed_categorization"].items():
            lines.append(f"  {cat}: {info['count']:,} ({info['pct_of_missed']}% of missed)")
    lines.append("-- CAUGHT group (means only) --")
    lines.extend(_summary_block(exp3["caught"]))
    lines.append("-- MISSED group (means only) --")
    lines.extend(_summary_block(exp3["missed"]))
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 4 -- TRANSLITERATION / SCRIPT ANALYSIS (India)")
    lines.append("=" * 78)
    lines.append(f"india_true_matches: {exp4['india_true_matches']:,}")
    lines.append(f"with unequal norm names: {exp4['india_true_matches_with_unequal_norm_names']:,} "
                 f"({exp4['unequal_pct_of_india']}%)")
    lines.append("script_pair_distribution (among unequal-name true matches):")
    for k, v in exp4["script_pair_distribution"].items():
        lines.append(f"  {k}: {v:,}")
    lines.append(f"cross_script (latin<->devanagari) count: {exp4['cross_script_latin_devanagari_count']:,} "
                 f"({exp4['cross_script_pct_of_unequal']}% of unequal-name matches)")
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 5 -- UNION BLOCKING RECALL SWEEP")
    lines.append("=" * 78)
    meta = exp5.get("_meta", {})
    lines.append(f"sample: {meta.get('s1_sample_size')} S1 entities, pool: {meta.get('candidate_pool_size'):,}, "
                 f"runtime: {meta.get('seconds')}s")
    for label, r in exp5.items():
        if label == "_meta":
            continue
        lines.append(f"  [{label}]")
        lines.append(f"      recall={r['recall_pct']}%  avg_candidates={r['avg_candidates_per_s1']}  "
                     f"p95={r['p95_candidates_per_s1']}  p99={r['p99_candidates_per_s1']}  "
                     f"max={r['max_candidates_per_s1']}  reduction={r['reduction_ratio_pct']}%")

    path = os.path.join(AUDIT_DIR, "experiments_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    audit.log(f"Summary written to {path}")


if __name__ == "__main__":
    main()
