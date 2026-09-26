#!/usr/bin/env python3
"""
Scaled port of the teammate's 4-method blocking approach
(src/blocking_eval_approach.ipynb), built for running against a much
larger S1 sample (default 100,000) than the notebook's original 1,000, as
a stepping stone toward the full ~2.2M-S1 dataset.

Faithfully reproduces the notebook's exact logic, thresholds, and
combination rules (extracted directly from its executable cells, not the
stale/incorrect parameter manifest in its final "save artifacts" cell --
see the NOTE below), with ONE deliberate deviation:

  DEVIATION: the notebook fits ONE TfidfVectorizer over ALL S2+S3 records
  with no country pre-filter (country filtering happens only at query
  time via an index lookup into the single fitted matrix). At full scale
  that means fitting one vectorizer's vocabulary over ~10M records'
  character n-grams -- a real memory risk, and contrary to the
  hardware-aware plan doc's own country-sharding guidance ("partition by
  normalized country first... process one country shard completely...
  before starting the next"). This script fits a SEPARATE, smaller
  TfidfVectorizer PER COUNTRY SHARD instead. This changes the TF-IDF
  vocabulary and IDF weights (computed per-country instead of globally)
  but not the underlying similarity concept (char n-gram cosine
  similarity within a country), and the country-restriction the notebook
  already enforces at query time means this shouldn't meaningfully change
  which candidates are found -- just makes memory bounded per shard.

Everything else -- normalization, all 4 blockers' exact parameters
(NAME_CHAR_MIN_SIMILARITY=0.35, NAME_FUZZY_THRESHOLD=0.55,
ADDRESS_OVERLAP_THRESHOLD=2, top-k limits, char n-gram range (2,5), the
per-blocker score semantics, the union-of-all-4 combination, and the
exact provenance schema (blocking_methods/num_blockers/max_block_score) --
is a direct port. NOTE on the notebook's own manifest cell: its final
"save artifacts" step recorded some parameter values (jaccard_threshold,
levenshtein_threshold, ngram_size=3) that do NOT match what the executable
blocker cells actually use -- this port trusts the real executable code
(char n-gram range (2,5), fuzzy threshold 0.55, char-tfidf similarity
floor 0.35), not that stale manifest.

Dead code intentionally NOT ported (confirmed unused in the original):
  - a second, unused TfidfVectorizer(analyzer="char_wb") built in the
    notebook but never queried
  - `from rapidfuzz.distance import Levenshtein` (imported, never called)

Run from student_resource/, using the project venv:
    ../.venv/Scripts/python.exe code/business_entity_resolution/src/scaled_blocking.py

Env vars:
    BLOCKING_N_S1              how many S1 entities to sample (default 100000)
    BLOCKING_SEED              sampling seed (default 42, same as export_10k_sample.py)
    BLOCKING_LOG_EVERY         progress print interval in S1 rows (default 1000)

Output:
    pipeline_output/scaled_blocking/candidate_pairs.tsv    (s1_entity_id, candidate_entity_id)
    pipeline_output/scaled_blocking/candidate_provenance.tsv
    pipeline_output/scaled_blocking/s1_sample.tsv           (the sampled S1 entities, so
                                                              downstream stages know the
                                                              required-id set)
    pipeline_output/scaled_blocking/blocking_report.json    (recall vs ground truth,
                                                              per-method + union stats,
                                                              timing)
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit
import dataset_cache
from pipeline_common import PIPELINE_OUTPUT_DIR, log

N_S1 = int(os.environ.get("BLOCKING_N_S1", "100000"))
SEED = int(os.environ.get("BLOCKING_SEED", "42"))
LOG_EVERY = int(os.environ.get("BLOCKING_LOG_EVERY", "1000"))

OUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "scaled_blocking")

# ---------------------------------------------------------------------------
# Exact parameters from the notebook (see module docstring for the
# manifest-vs-real-code discrepancy this port resolves in favor of the
# real code)
# ---------------------------------------------------------------------------

NAME_CHAR_TOP_K = 50
NAME_FUZZY_TOP_K = 50
NAME_TOKEN_TOP_K = 50
ADDRESS_TOP_K = 30
ADDRESS_OVERLAP_THRESHOLD = 2
NAME_CHAR_NGRAM_RANGE = (2, 5)
NAME_CHAR_MIN_DF = 1
NAME_CHAR_MIN_SIMILARITY = 0.35
NAME_FUZZY_THRESHOLD = 0.55
MIN_TOKEN_LENGTH = 3  # address tokens
NAME_TOKEN_MIN_LENGTH = 2  # name tokens (notebook hardcodes this separately from MIN_TOKEN_LENGTH)

WEAK_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "highway", "hwy", "boulevard", "blvd", "way", "place", "pl", "parkway", "pkwy",
    "unit", "suite", "ste", "apt", "apartment", "floor", "fl", "building", "bldg",
    "block", "district", "city", "state", "county", "india", "usa", "us",
}

LEGAL_SUFFIXES = {
    "limited", "ltd", "llc", "inc", "incorporated", "corp", "corporation",
    "company", "co", "private", "pvt", "plc", "llp", "lp",
}

# ---------------------------------------------------------------------------
# Normalization -- exact port of the notebook's normalize_text/tokenize/
# transliteration helper chain (uses anyascii, NOT audit.py's regex-based
# normalization -- these are deliberately different normalization schemes,
# see schema_contracts.py's note on the two approaches coexisting)
# ---------------------------------------------------------------------------

import re
import unicodedata


def is_latin_char(ch):
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def contains_non_latin(text):
    if not isinstance(text, str):
        return False
    return any(ch.isalpha() and not is_latin_char(ch) for ch in text)


def transliterate_text(text):
    if not isinstance(text, str) or not text:
        return ""
    return anyascii(text)


def normalize_text(text, transliterate=False):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    if transliterate:
        text = transliterate_text(text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split() if text else []


def transliterated_tokens(text):
    normalized = normalize_text(text, transliterate=False)
    result = []
    for token in tokenize(normalized):
        if contains_non_latin(token):
            token = transliterate_text(token)
        token = normalize_text(token, transliterate=False)
        if token:
            result.append(token)
    return result


def get_informative_address_tokens(address):
    tokens = transliterated_tokens(address)
    informative = set()
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in WEAK_ADDRESS_TOKENS:
            continue
        if not re.search(r"[a-z0-9]", token):
            continue
        informative.add(token)
    return informative


def get_name_tokens(name):
    return transliterated_tokens(name)


def get_name_core_tokens(name):
    return [t for t in get_name_tokens(name) if t not in LEGAL_SUFFIXES]


def get_name_char_text(name):
    return normalize_text(name, transliterate=True)


# ---------------------------------------------------------------------------
# Sampling: N_S1 entities uniformly at random from the full dataset,
# same pattern as export_10k_sample.py (singletons included -- this is a
# BLOCKING RECALL test, not a training-pair sample, so singletons matter
# just as much as matched entities for measuring real-world candidate
# coverage).
# ---------------------------------------------------------------------------

def sample_s1(n, seed):
    log(f"Sampling {n:,} S1 entities uniformly at random (seed={seed})...")
    s1_df = dataset_cache.load_normalized("train_source1")
    rng = np.random.RandomState(seed)
    all_ids = s1_df["entity_id"].to_numpy()
    sampled_ids = set(rng.choice(all_ids, size=min(n, len(all_ids)), replace=False))
    sample = s1_df[s1_df["entity_id"].isin(sampled_ids)].copy()
    log(f"  sampled {len(sample):,} S1 entities")
    return sample


def load_candidate_pool(countries_needed):
    """Load S2+S3 records restricted to the countries actually present in
    the S1 sample -- no reason to load/index candidate records for a
    country with zero S1 entities to match against."""
    log(f"Loading S2+S3 candidate pool for countries: {sorted(countries_needed)}...")
    t0 = time.time()
    frames = []
    n_chunks = 0
    for src in ("train_source2", "train_source3"):
        for chunk in dataset_cache.load_normalized_lazy(
            src, columns=["entity_id", "business_name", "business_address", "country"]
        ):
            n_chunks += 1
            filtered = chunk[chunk["country"].isin(countries_needed)]
            if len(filtered):
                filtered = filtered.copy()
                filtered["source"] = "S2" if src == "train_source2" else "S3"
                frames.append(filtered)
            if n_chunks % 10 == 0:
                log(f"  ...scanned {n_chunks} chunks ({round(time.time()-t0,1)}s elapsed)")
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "country", "source"])
    log(f"  loaded {len(pool):,} candidate records in {round(time.time()-t0,1)}s")
    return pool


# ---------------------------------------------------------------------------
# Build the per-country-shard structures the notebook's blockers query
# ---------------------------------------------------------------------------

def _prepare_fields_chunk(chunk_dict):
    """Worker: compute the 5 derived fields for one chunk of pool rows.
    Runs in a separate process -- this is the fix for a real performance
    problem found while testing: preparing these fields via plain
    .apply() across the full ~10.3M-row candidate pool (needed regardless
    of S1 sample size, since both countries in this dataset are always
    present) took so long with no progress signal that a first test run
    had to be killed after ~15 minutes with no visible progress. Each
    row's tokenization/transliteration is independent, so this is
    embarrassingly parallel -- split into chunks, dispatch to workers."""
    chunk = pd.DataFrame(chunk_dict)
    chunk["address_tokens"] = chunk["business_address"].apply(get_informative_address_tokens)
    chunk["name_tokens"] = chunk["business_name"].apply(get_name_tokens)
    chunk["name_core_tokens"] = chunk["business_name"].apply(get_name_core_tokens)
    chunk["name_char_text"] = chunk["business_name"].apply(get_name_char_text)
    chunk["country_norm"] = chunk["country"].apply(lambda x: normalize_text(x, transliterate=True))
    return chunk


def prepare_pool_fields_parallel(pool_df):
    log(f"Preparing candidate pool fields (tokens, char text, country_norm) "
        f"for {len(pool_df):,} rows, parallel across {audit.MAX_WORKERS} workers...")
    t0 = time.time()

    n_workers = audit.MAX_WORKERS
    chunk_size = max(1, -(-len(pool_df) // n_workers))  # ceil division
    chunks = [pool_df.iloc[i:i + chunk_size].to_dict("list") for i in range(0, len(pool_df), chunk_size)]
    log(f"  split into {len(chunks)} chunks of ~{chunk_size:,} rows each")

    results_by_index = {}
    with audit.make_executor() as ex:
        future_to_index = {ex.submit(_prepare_fields_chunk, c): i for i, c in enumerate(chunks)}
        n_done = 0
        for fut in audit.as_completed(future_to_index):
            results_by_index[future_to_index[fut]] = fut.result()
            n_done += 1
            log(f"  chunk {n_done}/{len(chunks)} done ({round(time.time()-t0,1)}s elapsed)")

    ordered = [results_by_index[i] for i in range(len(chunks))]
    result = pd.concat(ordered, ignore_index=True)
    log(f"Pool field preparation done in {round(time.time()-t0,1)}s")
    return result


def build_structures(pool_df):
    pool_df = prepare_pool_fields_parallel(pool_df)

    source_lookup = pool_df.set_index("entity_id").to_dict("index")

    log("Building address inverted index...")
    address_index = defaultdict(set)
    for entity_id, row in zip(pool_df["entity_id"], pool_df["address_tokens"]):
        for token in row:
            address_index[token].add(entity_id)

    log("Building name token indexes...")
    name_token_index = defaultdict(set)
    name_core_token_index = defaultdict(set)
    for entity_id, tokens, core_tokens in zip(
        pool_df["entity_id"], pool_df["name_tokens"], pool_df["name_core_tokens"]
    ):
        for token in set(tokens):
            if len(token) >= NAME_TOKEN_MIN_LENGTH:
                name_token_index[token].add(entity_id)
        for token in set(core_tokens):
            if len(token) >= NAME_TOKEN_MIN_LENGTH:
                name_core_token_index[token].add(entity_id)

    log("Building per-country RapidFuzz choice dicts...")
    country_name_choices = defaultdict(dict)
    for country, entity_id, char_text in zip(
        pool_df["country_norm"], pool_df["entity_id"], pool_df["name_char_text"]
    ):
        if country:
            country_name_choices[country][entity_id] = char_text

    # DEVIATION from the notebook (documented in the module docstring):
    # fit ONE TfidfVectorizer PER COUNTRY SHARD instead of one global
    # vectorizer over the whole pool, to bound memory at scale.
    log("Fitting per-country TF-IDF vectorizers (char n-grams 2-5)...")
    tfidf_by_country = {}
    for country, group in pool_df.groupby("country_norm"):
        if not country:
            continue
        texts = group["name_char_text"].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(
            analyzer="char", ngram_range=NAME_CHAR_NGRAM_RANGE,
            min_df=NAME_CHAR_MIN_DF, lowercase=False,
        )
        matrix = vec.fit_transform(texts)
        tfidf_by_country[country] = (vec, matrix, group["entity_id"].to_numpy())
    log(f"  fitted {len(tfidf_by_country)} country-shard TF-IDF vectorizers")

    return {
        "source_lookup": source_lookup,
        "address_index": address_index,
        "name_token_index": name_token_index,
        "name_core_token_index": name_core_token_index,
        "country_name_choices": country_name_choices,
        "tfidf_by_country": tfidf_by_country,
    }


# ---------------------------------------------------------------------------
# The four blockers -- exact port of retrieve_address_candidates,
# retrieve_name_token_candidates, retrieve_name_char_candidates,
# retrieve_name_fuzzy_candidates from the notebook.
# ---------------------------------------------------------------------------

def retrieve_address_candidates(s1_row, structures):
    s1_id, country = s1_row["entity_id"], s1_row["country_norm"]
    query_tokens = s1_row["address_tokens"]
    if not query_tokens:
        return []
    counts = Counter()
    for token in query_tokens:
        for entity_id in structures["address_index"].get(token, ()):
            counts[entity_id] += 1
    ranked = []
    for entity_id, overlap in counts.most_common():
        if len(ranked) >= ADDRESS_TOP_K:
            break
        candidate = structures["source_lookup"][entity_id]
        if candidate["country_norm"] != country:
            continue
        if overlap < ADDRESS_OVERLAP_THRESHOLD:
            continue
        ranked.append({"s1_entity_id": s1_id, "candidate_entity_id": entity_id,
                        "block_method": "address", "block_score": float(overlap)})
    return ranked


def retrieve_name_token_candidates(s1_row, structures):
    s1_id, country = s1_row["entity_id"], s1_row["country_norm"]
    query_tokens = set(s1_row["name_tokens"])
    query_core_tokens = set(s1_row["name_core_tokens"])
    if not query_tokens:
        return []
    counts = Counter()
    for token in query_tokens:
        if len(token) < NAME_TOKEN_MIN_LENGTH:
            continue
        for entity_id in structures["name_token_index"].get(token, ()):
            counts[entity_id] += 1
    for token in query_core_tokens:
        if len(token) < NAME_TOKEN_MIN_LENGTH:
            continue
        for entity_id in structures["name_core_token_index"].get(token, ()):
            counts[entity_id] += 1
    ranked = []
    for entity_id, overlap in counts.most_common():
        if len(ranked) >= NAME_TOKEN_TOP_K:
            break
        candidate = structures["source_lookup"][entity_id]
        if candidate["country_norm"] != country:
            continue
        ranked.append({"s1_entity_id": s1_id, "candidate_entity_id": entity_id,
                        "block_method": "name_token", "block_score": float(overlap)})
    return ranked


def retrieve_name_char_candidates_batch(s1_batch_df, structures):
    """Batched replacement for a per-row vec.transform()+cosine_similarity()
    loop.

    PERFORMANCE FIX: the original per-row version (kept in git history)
    computes ONE query vector and does ONE dense cosine_similarity() call
    against the full country TF-IDF matrix per S1 row. Measured directly:
    ~0.74s per call against a realistic ~500k-row matrix (so several
    seconds against a full ~6M-row country matrix), which alone would be
    on the order of days at the target 100k-S1 scale.

    Simply batching all of a country's S1 queries into one
    cosine_similarity() call is not a safe fix either -- that returns a
    DENSE (n_queries x n_candidates) array; at 100k queries x 6M
    candidates that's a ~4.8 TB dense matrix, nowhere close to fitting in
    memory. The actual fix is sparse_dot_topn's sp_matmul_topn(), the tool
    the hardware-aware plan doc itself calls out for exactly this problem:
    it computes the same top-N-per-row cosine similarities but keeps the
    result SPARSE (only the top_n nonzero entries per row are ever
    materialized), so memory scales with n_queries * top_n, not
    n_queries * n_candidates. Measured: ~4.2 minutes for 500 queries
    against a 6M-row matrix with a genuinely sparse result -- the same
    similarity values sp_matmul_topn and cosine_similarity would agree on
    for the entries that survive the top_n/threshold cut, just computed
    without ever forming the full dense matrix.

    Returns a list of result dicts across ALL rows in s1_batch_df (which
    must all share the same country_norm)."""
    results = []
    for country, group in s1_batch_df.groupby("country_norm"):
        entry = structures["tfidf_by_country"].get(country)
        if entry is None:
            continue
        vec, matrix, entity_ids = entry

        queries_df = group[group["name_char_text"] != ""]
        if queries_df.empty:
            continue
        query_ids = queries_df["entity_id"].tolist()
        query_vectors = vec.transform(queries_df["name_char_text"].tolist())

        # sp_matmul_topn wants both operands as sparse matrices with
        # matching inner dimension -- matrix is (n_candidates, n_features),
        # so transpose it to (n_features, n_candidates) for the multiply.
        sim_matrix = sp_matmul_topn(
            query_vectors, matrix.T, top_n=NAME_CHAR_TOP_K,
            threshold=NAME_CHAR_MIN_SIMILARITY, n_threads=-1,
        )
        sim_matrix = sim_matrix.tocsr()

        for row_idx, s1_id in enumerate(query_ids):
            row = sim_matrix.getrow(row_idx)
            for col_idx, score in zip(row.indices, row.data):
                results.append({
                    "s1_entity_id": s1_id, "candidate_entity_id": entity_ids[col_idx],
                    "block_method": "name_char_tfidf", "block_score": float(score),
                })
    return results


def retrieve_name_fuzzy_candidates_batch(s1_batch_df, structures):
    """Batched replacement for a per-row process.extract() loop.

    PERFORMANCE FIX: the original per-row version (kept in git history,
    functionally identical results) calls rapidfuzz's process.extract()
    once per S1 row against a country's full choices dict. Measured
    directly: ~1.5s per call against a realistic ~6M-entry country pool,
    which is ~12.6 minutes for just this blocker across 500 S1 rows, and
    would be ~42 HOURS at the target 100k-S1 scale -- confirmed by
    profiling after a test run stalled with no progress for 10+ minutes on
    only 500 rows. rapidfuzz's process.cdist() computes the full
    query-x-choice similarity matrix in one batched, multi-threaded C call
    instead of one Python-level call per query: measured 9.2s for 500
    queries against 6M choices (vs. ~12.6 minutes via extract() loop) --
    roughly an 80x speedup from batching alone, same scorer, same
    threshold, same top-k truncation, same results.

    Returns a list of result dicts across ALL rows in s1_batch_df (which
    must all share the same country_norm), not just one row -- the driver
    loop calls this once per country instead of once per S1 row."""
    results = []
    for country, group in s1_batch_df.groupby("country_norm"):
        if not country:
            continue
        choices = structures["country_name_choices"].get(country, {})
        if not choices:
            continue
        choice_ids = list(choices.keys())
        choice_texts = list(choices.values())

        queries_df = group[group["name_char_text"] != ""]
        if queries_df.empty:
            continue
        query_texts = queries_df["name_char_text"].tolist()
        query_ids = queries_df["entity_id"].tolist()

        score_matrix = process.cdist(query_texts, choice_texts, scorer=fuzz.ratio, workers=-1)

        # for each query row, keep only the top NAME_FUZZY_TOP_K scores,
        # then apply the same >= NAME_FUZZY_THRESHOLD (0.55, on a 0-1
        # scale after /100) filter as the original per-row version.
        for row_idx, s1_id in enumerate(query_ids):
            row_scores = score_matrix[row_idx]
            top_k_idx = np.argpartition(row_scores, -min(NAME_FUZZY_TOP_K, len(row_scores)))[-NAME_FUZZY_TOP_K:]
            for i in top_k_idx:
                normalized_score = float(row_scores[i]) / 100.0
                if normalized_score < NAME_FUZZY_THRESHOLD:
                    continue
                results.append({"s1_entity_id": s1_id, "candidate_entity_id": choice_ids[i],
                                 "block_method": "name_fuzzy", "block_score": normalized_score})
    return results


# ---------------------------------------------------------------------------
# Driver: run all 4 blockers, union + provenance, matching the notebook's
# cells 16-27's COMBINATION logic exactly. Execution strategy differs from
# the notebook (see the two _batch functions' docstrings for why): address
# and name_token are cheap dict lookups and stay per-row; name_char_tfidf
# and name_fuzzy are the two blockers that were measured to be
# catastrophically slow per-row at real country-pool scale (~42h and
# ~247h respectively, extrapolated, for just those two blockers at
# 100k S1), so those two run as one batched call PER COUNTRY instead of
# one call per S1 row -- same scorer, same thresholds, same top-k, same
# candidate results, just computed without the per-row Python-call
# overhead that made the naive port impractical.
# ---------------------------------------------------------------------------

def run_blocking(s1_sample, structures):
    s1_sample = s1_sample.copy()
    s1_sample["address_tokens"] = s1_sample["business_address"].apply(get_informative_address_tokens)
    s1_sample["name_tokens"] = s1_sample["business_name"].apply(get_name_tokens)
    s1_sample["name_core_tokens"] = s1_sample["business_name"].apply(get_name_core_tokens)
    s1_sample["name_char_text"] = s1_sample["business_name"].apply(get_name_char_text)
    s1_sample["country_norm"] = s1_sample["country"].apply(lambda x: normalize_text(x, transliterate=True))

    all_rows = []
    total = len(s1_sample)
    t0 = time.time()

    log("Running address + name_token blockers (per-row dict lookups)...")
    for i, (_, row) in enumerate(s1_sample.iterrows(), start=1):
        if i == 1 or i % LOG_EVERY == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate if rate else 0
            log(f"  Processing S1 {i:,}/{total:,} ({rate:.1f}/s, ETA {eta/60:.1f} min)")
        all_rows.extend(retrieve_address_candidates(row, structures))
        all_rows.extend(retrieve_name_token_candidates(row, structures))
    log(f"  done in {round(time.time()-t0,1)}s")

    log("Running name_char_tfidf blocker (batched per country)...")
    t1 = time.time()
    all_rows.extend(retrieve_name_char_candidates_batch(s1_sample, structures))
    log(f"  done in {round(time.time()-t1,1)}s")

    log("Running name_fuzzy blocker (batched per country)...")
    t2 = time.time()
    all_rows.extend(retrieve_name_fuzzy_candidates_batch(s1_sample, structures))
    log(f"  done in {round(time.time()-t2,1)}s")

    log(f"Blocking done in {round(time.time()-t0,1)}s total ({total} S1 rows)")

    all_df = pd.DataFrame(all_rows, columns=["s1_entity_id", "candidate_entity_id", "block_method", "block_score"])

    candidate_pairs = all_df[["s1_entity_id", "candidate_entity_id"]].drop_duplicates().reset_index(drop=True)

    provenance = (
        all_df.groupby(["s1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(
            blocking_methods=("block_method", lambda v: ",".join(sorted(set(v)))),
            num_blockers=("block_method", "nunique"),
            max_block_score=("block_score", "max"),
        )
    )

    per_method_counts = all_df["block_method"].value_counts().to_dict()
    return candidate_pairs, provenance, per_method_counts


# ---------------------------------------------------------------------------
# Recall measurement against real ground truth for the sampled S1 entities
# ---------------------------------------------------------------------------

def measure_recall(candidate_pairs, s1_sample):
    log("Measuring blocking recall against ground truth...")
    gt = audit.load_ground_truth_map(only_ids=set(s1_sample["entity_id"]))
    total_true = sum(len(v) for v in gt.values())

    cand_by_s1 = candidate_pairs.groupby("s1_entity_id")["candidate_entity_id"].apply(set)
    hits = 0
    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        found = cand_by_s1.get(s1_id, set())
        hits += len(true_matches & found)

    recall = hits / total_true if total_true else None
    log(f"  {hits:,} of {total_true:,} true pairs recovered "
        f"({100*recall:.2f}% recall)" if recall is not None else "  no true pairs to check")
    return recall, total_true, hits


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    s1_sample = sample_s1(N_S1, SEED)
    countries_needed = set(s1_sample["country"].unique())

    pool_df = load_candidate_pool(countries_needed)
    structures = build_structures(pool_df)

    candidate_pairs, provenance, per_method_counts = run_blocking(s1_sample, structures)

    log(f"Union candidate pairs: {len(candidate_pairs):,} "
        f"(avg {len(candidate_pairs)/len(s1_sample):.1f} candidates/S1)")
    log(f"Per-method raw row counts: {per_method_counts}")

    recall, total_true, hits = measure_recall(candidate_pairs, s1_sample)

    candidate_pairs_path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    provenance_path = os.path.join(OUT_DIR, "candidate_provenance.tsv")
    s1_sample_path = os.path.join(OUT_DIR, "s1_sample.tsv")

    candidate_pairs.to_csv(candidate_pairs_path, sep="\t", index=False)
    provenance.to_csv(provenance_path, sep="\t", index=False)
    s1_sample[["entity_id", "business_name", "business_address", "country"]].to_csv(
        s1_sample_path, sep="\t", index=False
    )
    log(f"Wrote {candidate_pairs_path}")
    log(f"Wrote {provenance_path}")
    log(f"Wrote {s1_sample_path}")

    report = {
        "n_s1_sampled": len(s1_sample),
        "seed": SEED,
        "countries": sorted(countries_needed),
        "candidate_pool_size": len(pool_df),
        "n_candidate_pairs": len(candidate_pairs),
        "avg_candidates_per_s1": round(len(candidate_pairs) / len(s1_sample), 2),
        "per_method_raw_row_counts": per_method_counts,
        "total_true_pairs": total_true,
        "true_pairs_recovered": hits,
        "recall_pct": round(100 * recall, 3) if recall is not None else None,
        "total_seconds": round(time.time() - t_start, 1),
        "deviation_from_notebook": "TF-IDF fitted per-country-shard, not globally "
                                   "(see module docstring)",
    }
    report_path = os.path.join(OUT_DIR, "blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log(f"\nTotal runtime: {report['total_seconds']}s")


if __name__ == "__main__":
    main()
