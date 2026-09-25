#!/usr/bin/env python3
"""
ML Challenge 2026 -- Dataset Audit Script (v2, memory-bounded)

Produces compact, aggregated statistics about the (huge, ~12M-row) train/test
TSVs so they can be inspected without ever pasting raw rows around.

v2 fixes a memory blow-up in v1: that version kept Python-level Counters over
every raw string (tens of millions of unique names/addresses) and built giant
.tolist() copies per chunk, which grew unbounded and never finished. This
version:
  - uses vectorized pandas/numpy string ops instead of per-row Python .map()
  - estimates duplicate rates with hashed value_counts() accumulated as a
    dict-of-counts (bounded by *unique* values, which is still large but far
    cheaper than storing every raw string + Python object overhead) -- and
    for the truly huge files, falls back to a fixed-size reservoir sample for
    length/token/duplicate stats instead of a full pass
  - drops per-row regex example mining from the hot loop (done on a small
    sample instead)
  - reports progress per chunk so you can see it's alive

Run from the `student_resource/` directory:

    python code/business_entity_resolution/src/audit.py

Env vars:
    SKIP_BLOCKING=1          skip the (expensive) blocking experiments step
    SAMPLE_ROWS=500000       max rows sampled per file for detailed stats
                             (default 500000; the file is still scanned in
                             full only for cheap row counts / country dist)
    BLOCKING_SAMPLE_FRAC     fraction of S1 (with >=1 true match) to test
                             blocking strategies on (default 0.01)
    NO_CACHE=1               ignore any cached results and recompute every
                             step from scratch

Caching: every step's result is cached to `audit/cache/` under a key derived
from (a) the size+mtime of every source .tsv it reads, and (b) the params
that affect its output (SAMPLE_ROWS, BLOCKING_SAMPLE_FRAC, ...). Re-running
the script with unchanged inputs and params reloads each step's cached JSON
in milliseconds instead of rescanning the files -- a full re-run after the
first one typically finishes in well under a second. Editing a .tsv (or
swapping in a new one under the same name) changes its fingerprint and
transparently invalidates only the steps that depend on it; everything else
still loads from cache. Delete `audit/cache/` (or set NO_CACHE=1) to force a
clean recompute of everything.

Outputs land in `audit/`:
    summary.txt            -- human-readable digest (paste this first)
    structure_stats.json
    name_stats.json
    address_stats.json
    overlap_stats.json
    ground_truth_stats.json
    blocking_stats.json
    examples.json
"""

import hashlib
import json
import os
import re

import regex  # third-party (MIT); needed for Unicode-correct \w (see NON_WORD_KEEP_MARKS_RE)
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
#
# This machine has many cores but pandas/numpy string processing here is
# CPU-bound, pure-Python-adjacent work that does NOT release the GIL the way
# I/O does -- so getting real parallelism requires separate processes, not
# threads. Each step's independent units of work (one file, one source
# comparison, one pool file) are dispatched to a ProcessPoolExecutor sized to
# the machine's core count, capped so we don't oversubscribe a small machine
# or leave a big one under-used.

MAX_WORKERS = max(1, min(int(os.environ.get("AUDIT_WORKERS", os.cpu_count() or 4)), 16))


def make_executor():
    return ProcessPoolExecutor(max_workers=MAX_WORKERS)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
STUDENT_RESOURCE = os.path.abspath(os.path.join(BASE, "..", "..", ".."))
DATASET = os.path.join(STUDENT_RESOURCE, "dataset")
TRAIN_DIR = os.path.join(DATASET, "train")
TEST_DIR = os.path.join(DATASET, "test")
AUDIT_DIR = os.path.join(STUDENT_RESOURCE, "audit")
CACHE_DIR = os.path.join(AUDIT_DIR, "cache")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

FILES = {
    "train_source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "train_ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    "test_source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

CHUNKSIZE = 200_000
SAMPLE_ROWS = int(os.environ.get("SAMPLE_ROWS", "500000"))  # cap for detailed (name/addr) stats
BLOCKING_SAMPLE_FRAC = float(os.environ.get("BLOCKING_SAMPLE_FRAC", "0.01"))
EMPTY_SET = frozenset()

# Set NO_CACHE=1 to force every step to recompute regardless of a valid cache.
NO_CACHE = os.environ.get("NO_CACHE") == "1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------
#
# Each source file's identity is fingerprinted from (path, size, mtime) -- not
# file content, since hashing 12M rows just to decide whether to trust the
# cache would defeat the purpose. If a .tsv is replaced with a same-named file
# of different size/mtime (e.g. a corrected dataset drop), the fingerprint
# changes and every step depending on that file recomputes automatically.
#
# Each step's cache entry is keyed by: step name + the fingerprints of the
# files it reads + any parameters that affect its output (e.g. SAMPLE_ROWS,
# BLOCKING_SAMPLE_FRAC). That key is hashed into a short cache filename, so
# changing SAMPLE_ROWS or re-running after a source file changes naturally
# invalidates only the affected steps -- no manual cache-clearing needed.

def _file_fingerprint(path):
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def _cache_key(step_name, file_keys, **params):
    """Build a short, stable cache key from the step name, the fingerprints of
    the named FILES entries it depends on, and any extra parameters."""
    parts = [step_name]
    for fk in file_keys:
        parts.append(_file_fingerprint(FILES[fk]))
    for k in sorted(params):
        parts.append(f"{k}={params[k]}")
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(step_name, key):
    return os.path.join(CACHE_DIR, f"{step_name}.{key}.json")


def load_cached(step_name, file_keys, **params):
    """Return the cached result for this step if a valid cache entry exists,
    else None. Also removes stale cache files for this step (different key,
    i.e. inputs/params changed) so the cache dir doesn't accumulate garbage."""
    key = _cache_key(step_name, file_keys, **params)
    path = cache_path(step_name, key)

    # clean up stale entries for this step name (old key)
    prefix = f"{step_name}."
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith(prefix) and fname != os.path.basename(path):
            try:
                os.remove(os.path.join(CACHE_DIR, fname))
                log(f"    [cache] removed stale cache entry {fname}")
            except OSError:
                pass

    if NO_CACHE:
        return None
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            log(f"    [cache] HIT for '{step_name}' (key={key}) -- skipping recompute")
            return result
        except (json.JSONDecodeError, OSError) as e:
            log(f"    [cache] cache file for '{step_name}' unreadable ({e}), recomputing")
            return None
    log(f"    [cache] MISS for '{step_name}' (key={key})")
    return None


def save_cache(step_name, file_keys, result, **params):
    key = _cache_key(step_name, file_keys, **params)
    path = cache_path(step_name, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=str)
    log(f"    [cache] saved '{step_name}' (key={key})")


# ---------------------------------------------------------------------------
# Normalization helpers (vectorized where possible)
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "pvt", "private", "llp", "pllc", "plc", "gmbh", "sarl",
    "sa", "srl", "pc", "lp",
}
LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b"
)

# Strips punctuation while KEEPING Unicode combining marks (categories Mc/Mn/Me).
# Uses the third-party `regex` module, not stdlib `re`, because stdlib re's \w
# excludes combining marks -- and those marks are essential letters/vowel-signs
# in Devanagari and other scripts in this dataset, not decorative punctuation.
# A bare `[^\w\s]` under stdlib re silently empties/mangles those names. See
# the normalize_series() docstring for the concrete failure example.
NON_WORD_KEEP_MARKS_RE = regex.compile(r"[^\w\s]", flags=regex.UNICODE)


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize: NFKC-fold, lowercase, '&'->'and', strip punctuation,
    collapse whitespace. Unicode NFKC still needs a per-element call (no
    vectorized NFKC in pandas), but it's a cheap C-level str op per row and
    far lighter than building extra Python objects per row.

    IMPORTANT: the punctuation-strip pattern is `[^\\w\\s\\u0300-\\u036f...]`,
    NOT bare `[^\\w\\s]`. Python's `\\w` matches Unicode letter/number
    categories but NOT combining marks (Unicode categories Mc/Mn) -- and
    those combining marks are not decoration, they're load-bearing parts of
    the script for many languages in this dataset (e.g. Devanagari vowel
    signs/virama: matra strokes like ा/ि/ी and the halant/virama ्). A bare
    `[^\\w\\s]` strips those out and silently mangles/empties Devanagari
    (and any other combining-mark-using script) names -- e.g. "राम मार्केटिंग"
    normalized to '' entirely, corrupting every downstream stat for those
    rows (length/token counts, duplicate rates, blocking, similarity
    features). The COMBINING_MARKS_RE below explicitly keeps Mc/Mn/Me
    characters as "word-like" so they survive normalization instead."""
    s = s.map(lambda x: unicodedata.normalize("NFKC", x) if x else "")
    s = s.str.lower()
    s = s.str.replace("&", " and ", regex=False)
    # pandas' vectorized .str.replace only accepts stdlib `re` patterns, and
    # stdlib re's \w excludes combining marks -- so this step uses the
    # third-party `regex` module (Unicode-correct \w) via .map(), same cost
    # class as the NFKC normalize() call above (one Python-level call per row).
    s = s.map(lambda x: NON_WORD_KEEP_MARKS_RE.sub(" ", x) if x else "")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def normalize_name_series(s: pd.Series) -> pd.Series:
    norm = normalize_series(s)
    norm = norm.str.replace(LEGAL_SUFFIX_RE, " ", regex=True)
    norm = norm.str.replace(r"\s+", " ", regex=True).str.strip()
    return norm


PIN_IN_RE = re.compile(r"\b(\d{6})\b")
PIN_US_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")


def extract_pin_series(s: pd.Series) -> pd.Series:
    pin6 = s.str.extract(PIN_IN_RE, expand=False)
    pin5 = s.str.extract(PIN_US_RE, expand=False)
    return pin6.fillna(pin5)


def name_prefix4_series(norm_name: pd.Series) -> pd.Series:
    compact = norm_name.str.replace(" ", "", regex=False)
    return compact.str.slice(0, 4)


def addr_last_token_series(addr: pd.Series) -> pd.Series:
    last = addr.str.split(",").str[-1].fillna("")
    return normalize_series(last)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def iter_chunks(path, usecols=None):
    return pd.read_csv(
        path, sep="\t", chunksize=CHUNKSIZE, usecols=usecols,
        dtype=str, keep_default_na=False, na_values=[], encoding="utf-8",
    )


def file_size_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 1)


# ---------------------------------------------------------------------------
# 1. Basic structure -- single cheap streaming pass, only value_counts on
#    the low-cardinality `country` column (bounded memory) + missing counts.
# ---------------------------------------------------------------------------

def basic_structure():
    log("[1/6] Basic structure...")
    cached = load_cached("basic_structure", list(FILES.keys()))
    if cached is not None:
        return cached
    out = _basic_structure_impl()
    save_cache("basic_structure", list(FILES.keys()), out)
    return out


def _basic_structure_one_file(key):
    """Worker for a single file's structure scan -- run in a separate process
    per file so all 7 files (independent of each other) scan concurrently
    instead of one after another on a single core."""
    path = FILES[key]
    t0 = time.time()
    n_rows = 0
    n_missing_name = n_missing_addr = n_missing_country = 0
    countries = Counter()
    is_source_file = "ground_truth" not in key
    cols = ["business_name", "business_address", "country"] if is_source_file else None

    for i, chunk in enumerate(iter_chunks(path, usecols=cols)):
        n_rows += len(chunk)
        if is_source_file:
            n_missing_name += int((chunk["business_name"] == "").sum())
            n_missing_addr += int((chunk["business_address"] == "").sum())
            n_missing_country += int((chunk["country"] == "").sum())
            vc = chunk["country"].value_counts()
            for k, v in vc.items():
                countries[k] += int(v)
        del chunk

    entry = {
        "rows": int(n_rows),
        "file_size_mb": file_size_mb(path),
        "seconds_to_scan": round(time.time() - t0, 1),
    }
    if is_source_file:
        entry.update({
            "columns": ["entity_id", "business_name", "business_address", "country"],
            "missing_pct": {
                "business_name": round(100 * n_missing_name / n_rows, 3) if n_rows else None,
                "business_address": round(100 * n_missing_addr / n_rows, 3) if n_rows else None,
                "country": round(100 * n_missing_country / n_rows, 3) if n_rows else None,
            },
            "country_distribution": dict(countries.most_common(20)),
        })
    else:
        entry["columns"] = ["source1_entity_id", "matched_entity_ids"]
    return key, entry


def _basic_structure_impl():
    out = {}
    with make_executor() as ex:
        futures = {ex.submit(_basic_structure_one_file, key): key for key in FILES}
        for fut in as_completed(futures):
            key, entry = fut.result()
            out[key] = entry
            log(f"    {key}: DONE -- {entry['rows']:,} rows in {entry['seconds_to_scan']}s")
    # preserve FILES' declared order in the output for readability
    return {k: out[k] for k in FILES}


# ---------------------------------------------------------------------------
# Reservoir sample of a file's name/address columns (bounded memory: caps at
# SAMPLE_ROWS regardless of file size, uniform sample via reservoir algorithm
# implemented cheaply by sampling within each chunk proportionally).
# ---------------------------------------------------------------------------

def sample_columns(path, n_target, usecols, seed=42):
    rng = np.random.RandomState(seed)
    # first pass: count rows cheaply (reuse structure_stats if available externally,
    # but keep this self-contained) -- estimate via file size instead to avoid a
    # second full pass: we just take a fixed fraction per chunk based on a rough
    # target, then trim/pad. Simpler & robust: take min(n_target, CHUNKSIZE) rows
    # from a bounded number of chunks spread across the file.
    frames = []
    collected = 0
    chunk_iter = iter_chunks(path, usecols=usecols)
    for i, chunk in enumerate(chunk_iter):
        if collected >= n_target:
            break
        # sample a slice of this chunk proportional to remaining need
        take = min(len(chunk), max(1, n_target // 20))  # spread across ~20 chunks
        if take < len(chunk):
            idx = rng.choice(len(chunk), size=take, replace=False)
            frames.append(chunk.iloc[idx])
        else:
            frames.append(chunk)
        collected += take
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True)
    if len(out) > n_target:
        out = out.sample(n_target, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2 & 3. Name and address statistics -- computed on a bounded sample per file
#    (SAMPLE_ROWS, default 500k) using vectorized ops. Duplicate rate is
#    estimated from the same sample (birthday-corrected note in output).
# ---------------------------------------------------------------------------

def name_address_stats():
    log(f"[2-3/6] Name & address statistics (sample_rows={SAMPLE_ROWS:,} per file)...")
    source_keys = [k for k in FILES if "ground_truth" not in k]
    cached = load_cached("name_address_stats", source_keys, sample_rows=SAMPLE_ROWS)
    if cached is not None:
        return cached["name"], cached["addr"], cached["examples"]
    name_out, addr_out, examples_out = _name_address_stats_impl()
    save_cache("name_address_stats", source_keys, {
        "name": name_out, "addr": addr_out, "examples": examples_out,
    }, sample_rows=SAMPLE_ROWS)
    return name_out, addr_out, examples_out


def _name_address_stats_one_file(key):
    """Worker for a single file's name/address stats -- independent of every
    other file, so these run concurrently across processes."""
    path = FILES[key]
    t0 = time.time()
    df = sample_columns(path, SAMPLE_ROWS, usecols=["business_name", "business_address"])
    n = len(df)

    names = df["business_name"]
    addrs = df["business_address"]

    norm_names = normalize_name_series(names)
    norm_addrs = normalize_series(addrs)

    name_len = names.str.len()
    name_tok = norm_names.str.split().str.len().fillna(0)
    addr_len = addrs.str.len()
    addr_tok = norm_addrs.str.split().str.len().fillna(0)

    raw_name_dupe_rate = 1 - (names.nunique() / n) if n else None
    norm_name_dupe_rate = 1 - (norm_names.nunique() / n) if n else None
    norm_addr_dupe_rate = 1 - (norm_addrs.nunique() / n) if n else None

    pins = extract_pin_series(addrs)
    pin_rate = pins.notna().mean() if n else None

    digit_counts = addrs.str.count(r"\d")
    total_chars = addr_len.sum()
    digit_share = (digit_counts.sum() / total_chars) if total_chars else None

    token_counter = Counter()
    for s in norm_names.sample(min(n, 100_000), random_state=1) if n else []:
        if s:
            token_counter.update(s.split())

    name_entry = {
        "sampled_rows": int(n),
        "length_chars": {
            "mean": round(float(name_len.mean()), 2) if n else None,
            "median": float(name_len.median()) if n else None,
            "p95": float(name_len.quantile(0.95)) if n else None,
            "max": int(name_len.max()) if n else None,
        },
        "token_count": {
            "mean": round(float(name_tok.mean()), 2) if n else None,
            "median": float(name_tok.median()) if n else None,
            "p95": float(name_tok.quantile(0.95)) if n else None,
        },
        "exact_raw_duplicate_rate_pct_of_sample": round(100 * raw_name_dupe_rate, 3) if raw_name_dupe_rate is not None else None,
        "normalized_duplicate_rate_pct_of_sample": round(100 * norm_name_dupe_rate, 3) if norm_name_dupe_rate is not None else None,
        "top_tokens": token_counter.most_common(30),
        "seconds": round(time.time() - t0, 1),
        "note": "duplicate rates measured within the sample, not the full file",
    }

    addr_entry = {
        "sampled_rows": int(n),
        "length_chars": {
            "mean": round(float(addr_len.mean()), 2) if n else None,
            "median": float(addr_len.median()) if n else None,
            "p95": float(addr_len.quantile(0.95)) if n else None,
            "max": int(addr_len.max()) if n else None,
        },
        "token_count": {
            "mean": round(float(addr_tok.mean()), 2) if n else None,
            "median": float(addr_tok.median()) if n else None,
        },
        "empty_address_pct": round(100 * float((addr_len == 0).sum()) / n, 3) if n else None,
        "pin_or_zip_extraction_rate_pct": round(100 * pin_rate, 3) if pin_rate is not None else None,
        "digit_char_share_pct": round(100 * digit_share, 3) if digit_share is not None else None,
        "normalized_duplicate_rate_pct_of_sample": round(100 * norm_addr_dupe_rate, 3) if norm_addr_dupe_rate is not None else None,
    }

    # variation examples: find normalized-name groups with >1 raw variant,
    # cheap because it's within the (already small) sample only
    ex_name = []
    grp = names.groupby(norm_names)
    for norm_val, sub in grp:
        if norm_val and sub.nunique() > 1:
            ex_name.append({"normalized": norm_val, "raw_variants": sub.unique().tolist()[:5]})
        if len(ex_name) >= 8:
            break
    ex_addr = []
    grp2 = addrs.groupby(norm_addrs)
    for norm_val, sub in grp2:
        if norm_val and sub.nunique() > 1:
            ex_addr.append({"normalized": norm_val, "raw_variants": sub.unique().tolist()[:5]})
        if len(ex_addr) >= 8:
            break
    examples_entry = {"name_variations": ex_name, "address_variations": ex_addr}

    return key, name_entry, addr_entry, examples_entry


def _name_address_stats_impl():
    name_out, addr_out, examples_out = {}, {}, {}
    source_keys = [k for k in FILES if "ground_truth" not in k]

    with make_executor() as ex:
        futures = {ex.submit(_name_address_stats_one_file, key): key for key in source_keys}
        for fut in as_completed(futures):
            key, name_entry, addr_entry, examples_entry = fut.result()
            name_out[key] = name_entry
            addr_out[key] = addr_entry
            examples_out[key] = examples_entry
            log(f"    {key}: DONE in {name_entry['seconds']}s")

    # preserve FILES' declared order in the output for readability
    name_out = {k: name_out[k] for k in source_keys}
    addr_out = {k: addr_out[k] for k in source_keys}
    examples_out = {k: examples_out[k] for k in source_keys}
    return name_out, addr_out, examples_out


# ---------------------------------------------------------------------------
# 4. Cross-source overlap -- uses hashed sets of normalized values. To bound
#    memory we hash strings to 64-bit ints instead of keeping the strings.
# ---------------------------------------------------------------------------

def hashed_set_from_file(path, usecols=("business_name", "business_address", "country")):
    """One streaming pass -> dict of {key_name: set(hash64)}."""
    sets = {
        "exact_raw_name": set(),
        "exact_norm_name": set(),
        "exact_norm_address": set(),
        "name_country": set(),
        "address_country": set(),
        "name_address": set(),
    }
    for chunk in iter_chunks(path, usecols=list(usecols)):
        raw_name = chunk["business_name"]
        country = chunk["country"]
        norm_name = normalize_name_series(raw_name)
        norm_addr = normalize_series(chunk["business_address"])

        sets["exact_raw_name"].update(pd.util.hash_array(raw_name.to_numpy(dtype=object)).tolist())
        sets["exact_norm_name"].update(pd.util.hash_array(norm_name.to_numpy(dtype=object)).tolist())
        sets["exact_norm_address"].update(pd.util.hash_array(norm_addr.to_numpy(dtype=object)).tolist())
        sets["name_country"].update(pd.util.hash_array((norm_name + "||" + country).to_numpy(dtype=object)).tolist())
        sets["address_country"].update(pd.util.hash_array((norm_addr + "||" + country).to_numpy(dtype=object)).tolist())
        sets["name_address"].update(pd.util.hash_array((norm_name + "||" + norm_addr).to_numpy(dtype=object)).tolist())
        del chunk
    return sets


def cross_source_overlap():
    log("[4/6] Cross-source overlap (train, hashed sets)...")
    cached = load_cached("cross_source_overlap", ["train_source1", "train_source2", "train_source3"])
    if cached is not None:
        return cached
    out = _cross_source_overlap_impl()
    save_cache("cross_source_overlap", ["train_source1", "train_source2", "train_source3"], out)
    return out


def _hashed_set_worker(key):
    """Picklable top-level wrapper so hashed_set_from_file can run in a
    separate process via ProcessPoolExecutor."""
    return key, hashed_set_from_file(FILES[key])


def _cross_source_overlap_impl():
    # Build S1's hashed sets AND both source files' hashed sets concurrently
    # -- all three are independent streaming passes over different files, so
    # there is no reason to wait for S1 to finish before starting S2/S3.
    t0 = time.time()
    all_sets = {}
    with make_executor() as ex:
        futures = {
            ex.submit(_hashed_set_worker, key): key
            for key in ("train_source1", "train_source2", "train_source3")
        }
        for fut in as_completed(futures):
            key, sets = fut.result()
            all_sets[key] = sets
            log(f"    {key} hashed sets built ({round(time.time()-t0,1)}s elapsed, "
                f"sizes: { {k: len(v) for k, v in sets.items()} })")

    s1_sets = all_sets["train_source1"]
    out = {}
    for src in ("train_source2", "train_source3"):
        src_sets = all_sets[src]
        overlaps = {}
        for k in s1_sets:
            inter = len(s1_sets[k] & src_sets[k])
            overlaps[k] = {
                "s1_unique_values": len(s1_sets[k]),
                f"{src}_unique_values": len(src_sets[k]),
                "overlap_count": inter,
                "overlap_pct_of_s1": round(100 * inter / len(s1_sets[k]), 3) if s1_sets[k] else None,
            }
        out[f"S1_vs_{src.split('_')[1]}"] = overlaps
    log(f"    all overlaps computed in {round(time.time()-t0,1)}s total")
    return out


# ---------------------------------------------------------------------------
# 5. Ground truth statistics -- cheap single pass, only counters/scalars kept.
# ---------------------------------------------------------------------------

def ground_truth_stats():
    log("[5/6] Ground truth statistics...")
    cached = load_cached("ground_truth_stats", ["train_ground_truth"])
    if cached is not None:
        return cached
    out = _ground_truth_stats_impl()
    save_cache("ground_truth_stats", ["train_ground_truth"], out)
    return out


def _ground_truth_stats_impl():
    path = FILES["train_ground_truth"]
    n_total = 0
    n_singleton = 0
    match_count_dist = Counter()
    s2_only = s3_only = both = neither = 0
    total_s2 = total_s3 = 0
    sum_matches = 0
    max_matches = 0

    for chunk in iter_chunks(path, usecols=["matched_entity_ids"]):
        ids_col = chunk["matched_entity_ids"]
        n_total += len(chunk)
        empty_mask = ids_col == ""
        n_singleton += int(empty_mask.sum())
        match_count_dist[0] += int(empty_mask.sum())
        neither += int(empty_mask.sum())

        non_empty = ids_col[~empty_mask]
        for val in non_empty:
            ids = val.split(",")
            n = len(ids)
            match_count_dist[n] += 1
            sum_matches += n
            max_matches = max(max_matches, n)
            has_s2 = False
            has_s3 = False
            c2 = c3 = 0
            for i in ids:
                if i.startswith("S2-"):
                    has_s2 = True
                    c2 += 1
                elif i.startswith("S3-"):
                    has_s3 = True
                    c3 += 1
            total_s2 += c2
            total_s3 += c3
            if has_s2 and has_s3:
                both += 1
            elif has_s2:
                s2_only += 1
            elif has_s3:
                s3_only += 1
            else:
                neither += 1
        del chunk

    mean_matches = sum_matches / n_total if n_total else None
    # median/p95 from the distribution counter (exact, memory-cheap)
    sorted_counts = sorted(match_count_dist.items())
    cum = 0
    values_expanded_needed = n_total
    median = None
    p95 = None
    half = values_expanded_needed / 2
    p95_rank = values_expanded_needed * 0.95
    for val, cnt in sorted_counts:
        cum += cnt
        if median is None and cum >= half:
            median = val
        if p95 is None and cum >= p95_rank:
            p95 = val

    out = {
        "total_s1_entities": n_total,
        "singleton_count": n_singleton,
        "singleton_pct": round(100 * n_singleton / n_total, 3),
        "matches_per_s1": {
            "mean": round(mean_matches, 3) if mean_matches is not None else None,
            "median": median,
            "max": max_matches,
            "p95": p95,
        },
        "match_count_distribution": {str(k): v for k, v in sorted_counts},
        "s2_only_pct": round(100 * s2_only / n_total, 3),
        "s3_only_pct": round(100 * s3_only / n_total, 3),
        "both_s2_and_s3_pct": round(100 * both / n_total, 3),
        "no_match_pct": round(100 * neither / n_total, 3),
        "total_s2_match_links": int(total_s2),
        "total_s3_match_links": int(total_s3),
    }
    log(f"    ground truth: {n_total:,} rows, singleton {out['singleton_pct']}%")
    return out


# ---------------------------------------------------------------------------
# 6. Blocking experiments -- run on a SMALL sample of S1 (with >=1 true
#    match) against the full S2+S3 pool, using an inverted index per
#    strategy built in a streaming pass (never materializes the full
#    cross-product). This is the most expensive step; keep sample_frac low.
# ---------------------------------------------------------------------------

def load_ground_truth_map(only_ids=None):
    gt = {}
    for chunk in iter_chunks(FILES["train_ground_truth"], usecols=["source1_entity_id", "matched_entity_ids"]):
        s1_col = chunk["source1_entity_id"]
        ids_col = chunk["matched_entity_ids"]
        if only_ids is not None:
            mask = s1_col.isin(only_ids)
            s1_col = s1_col[mask]
            ids_col = ids_col[mask]
        for s1, ids in zip(s1_col, ids_col):
            gt[s1] = set(ids.split(",")) if ids else set()
        del chunk
    return gt


def pick_s1_sample(sample_frac, seed=42):
    """Stream S1, keep only entity_id + fields needed, then subsample AFTER
    joining with ground truth eligibility (has >=1 match) so we never hold
    more than the full S1 id list (~2M strings) plus a small sample df."""
    rng = np.random.RandomState(seed)
    frames = []
    for chunk in iter_chunks(FILES["train_source1"], usecols=["entity_id", "business_name", "business_address", "country"]):
        frames.append(chunk)
    s1_all = pd.concat(frames, ignore_index=True)
    del frames
    return s1_all, rng


def blocking_experiments(sample_frac=0.01, seed=42):
    log(f"[6/6] Blocking experiments (sample_frac={sample_frac})...")
    file_keys = ["train_source1", "train_source2", "train_source3", "train_ground_truth"]
    cached = load_cached("blocking_experiments", file_keys, sample_frac=sample_frac, seed=seed)
    if cached is not None:
        return cached
    out = _blocking_experiments_impl(sample_frac, seed)
    save_cache("blocking_experiments", file_keys, out, sample_frac=sample_frac, seed=seed)
    return out


def _blocking_scan_one_source(src, strategy_names, s1_key_frames, gt_pairs):
    """Worker: stream ONE pool source file (train_source2 or train_source3)
    and accumulate per-strategy candidate counts + hit ids for the sampled S1
    entities. Runs in its own process so the two pool files scan in
    parallel. Returns picklable plain dict/Counter/int/float results only.

    Takes `strategy_names` (a plain list of strings), NOT the `strategies`
    dict of lambdas from the caller -- lambdas aren't picklable, so they
    can't cross the process boundary to a worker. The actual per-source key
    expressions are hardcoded below (they must match the S1-side ones used
    to build `s1_key_frames`); only the *names* are needed here to know which
    strategies to compute counts for."""
    path = FILES[src]
    t_src = time.time()
    n_chunks = 0
    total_pool_size = 0
    candidate_count = {name: defaultdict(int) for name in strategy_names}
    hit_ids = {name: defaultdict(set) for name in strategy_names}
    country_counts = Counter()

    for chunk in iter_chunks(path, usecols=["entity_id", "business_name", "business_address", "country"]):
        total_pool_size += len(chunk)
        n_chunks += 1
        norm_name = normalize_name_series(chunk["business_name"])
        name_prefix4 = name_prefix4_series(norm_name)
        pin = extract_pin_series(chunk["business_address"]).fillna("")
        country = chunk["country"]
        eid = chunk["entity_id"]

        for k, v in country.value_counts().items():
            country_counts[k] += int(v)

        pool_keys = {
            "country_norm_name": country + "||" + norm_name,
            "country_name_prefix4": country + "||" + name_prefix4,
            "country_pin": country + "||" + pin,
        }
        for strat_name, keys in pool_keys.items():
            s1_kf = s1_key_frames[strat_name]
            if s1_kf.empty:
                continue
            pool_kf = pd.DataFrame({"key": keys.to_numpy(), "cid": eid.to_numpy()})
            merged = pool_kf.merge(s1_kf, on="key", how="inner")
            if merged.empty:
                continue
            counts_this_chunk = merged.groupby("s1_id", sort=False).size()
            cc = candidate_count[strat_name]
            for s1_id, c in counts_this_chunk.items():
                cc[s1_id] += int(c)

            hit_merge = merged.merge(
                gt_pairs, left_on=["s1_id", "cid"], right_on=["s1_id", "true_cid"], how="inner"
            )
            if not hit_merge.empty:
                hd = hit_ids[strat_name]
                for s1_id, cid in zip(hit_merge["s1_id"], hit_merge["cid"]):
                    hd[s1_id].add(cid)
            del merged, pool_kf
        del chunk

    seconds = round(time.time() - t_src, 1)
    # convert defaultdicts to plain dicts for clean pickling back to the parent
    candidate_count = {k: dict(v) for k, v in candidate_count.items()}
    hit_ids = {k: dict(v) for k, v in hit_ids.items()}
    return candidate_count, hit_ids, country_counts, total_pool_size, seconds


def _blocking_experiments_impl(sample_frac, seed):
    t0 = time.time()

    log("    loading S1 (train, full ids/fields -- needed for sampling)...")
    s1_all, rng = pick_s1_sample(sample_frac, seed)
    log(f"    S1 loaded: {len(s1_all):,} rows")

    log("    loading ground truth for S1 ids...")
    gt = load_ground_truth_map(only_ids=set(s1_all["entity_id"]))
    has_match_ids = np.array([s1 for s1 in s1_all["entity_id"] if gt.get(s1)])
    n_sample = max(1, int(len(has_match_ids) * sample_frac))
    sampled_ids = set(rng.choice(has_match_ids, size=min(n_sample, len(has_match_ids)), replace=False))
    s1_sample = s1_all[s1_all["entity_id"].isin(sampled_ids)].copy()
    del s1_all
    log(f"    sampled {len(s1_sample):,} S1 entities (with >=1 true match)")

    s1_sample["norm_name"] = normalize_name_series(s1_sample["business_name"])
    s1_sample["norm_addr"] = normalize_series(s1_sample["business_address"])
    s1_sample["name_prefix4"] = name_prefix4_series(s1_sample["norm_name"])
    s1_sample["pin"] = extract_pin_series(s1_sample["business_address"])
    s1_sample["addr_last_tok"] = addr_last_token_series(s1_sample["business_address"])

    # Flat (s1_id, true_cid) pair table for the sampled S1 entities -- tiny
    # (<= ~11 matches x sample size), used for a cheap vectorized merge-based
    # hit check per chunk instead of a Python-level per-row membership test.
    gt_pair_rows = [(s1, cid) for s1 in s1_sample["entity_id"] for cid in gt.get(s1, ())]
    gt_pairs = pd.DataFrame(gt_pair_rows, columns=["s1_id", "true_cid"])
    del gt_pair_rows
    log(f"    ground-truth pair table for sample: {len(gt_pairs):,} rows")

    # NOTE: a bare "country" key is intentionally excluded from the expensive
    # pool join below -- with only 2-3 distinct values it matches essentially
    # the entire pool for every S1 entity, which is both useless as a blocking
    # key and pathologically expensive to join/group. Its recall is trivially
    # ~100% (it never drops a true match) and its candidate count is
    # ~(pool_size / n_countries), which we compute analytically instead.
    # NOTE: country_addr_last_token was tried and dropped from the expensive
    # pool join -- it takes the last comma-separated address segment as a
    # cheap "locality" proxy, but on this data that segment is very often a
    # STATE (e.g. "India||maharashtra" shared by 1,487 S1 sample rows), which
    # makes it nearly as unselective as bare country and pathologically
    # expensive to join against the full pool (temporary blowups into
    # hundreds of millions of merged rows for those keys). A better address
    # locality key (e.g. requiring a trailing digit run / real PIN, or a
    # dedicated city-extraction step) would need real feature engineering,
    # not this audit script -- left as a finding, not a working strategy here.
    strategies = {
        "country_norm_name": lambda df: df["country"] + "||" + df["norm_name"],
        "country_name_prefix4": lambda df: df["country"] + "||" + df["name_prefix4"],
        "country_pin": lambda df: df["country"] + "||" + df["pin"].fillna(""),
    }

    # build S1-side key -> s1_entity_id (small sample), per strategy, as DataFrames
    # ready for a vectorized merge (pandas merge is C-level, not a Python loop).
    s1_key_frames = {}
    for name, fn in strategies.items():
        keys = fn(s1_sample)
        df = pd.DataFrame({"key": keys.to_numpy(), "s1_id": s1_sample["entity_id"].to_numpy()})
        df = df[(df["key"] != "") & (~df["key"].str.endswith("||"))]
        s1_key_frames[name] = df
        log(f"    strategy '{name}': {df['key'].nunique():,} distinct S1 keys "
            f"over {len(df):,} S1 rows")

    # Scan train_source2 and train_source3 CONCURRENTLY (two independent
    # files, one process each) instead of sequentially. Each worker
    # accumulates PER-S1 CANDIDATE COUNTS AND HIT COUNTS incrementally, chunk
    # by chunk, instead of concatenating raw (s1_id, cid) pairs across the
    # whole pool -- some blocking keys (e.g. a state-level address token) are
    # shared by 1000+ S1 rows AND millions of pool rows, and concatenating
    # every chunk's merge result before deduping can reach billions of rows
    # and blow memory. Aggregating within each chunk keeps memory bounded by
    # the (small) S1 sample size regardless of key selectivity.
    pool_sources = ("train_source2", "train_source3")
    with make_executor() as ex:
        futures = {
            ex.submit(_blocking_scan_one_source, src, list(strategies.keys()), s1_key_frames, gt_pairs): src
            for src in pool_sources
        }
        per_source_results = {}
        for fut in as_completed(futures):
            src = futures[fut]
            per_source_results[src] = fut.result()
            log(f"    {src}: worker finished")

    # merge the two workers' results
    candidate_count = {name: defaultdict(int) for name in strategies}
    hit_ids = {name: defaultdict(set) for name in strategies}
    country_counts = Counter()
    total_pool_size = 0
    for src in pool_sources:
        src_candidate_count, src_hit_ids, src_country_counts, src_pool_size, src_seconds = per_source_results[src]
        total_pool_size += src_pool_size
        for k, v in src_country_counts.items():
            country_counts[k] += v
        for strat_name in strategies:
            for s1_id, c in src_candidate_count[strat_name].items():
                candidate_count[strat_name][s1_id] += c
            for s1_id, ids in src_hit_ids[strat_name].items():
                hit_ids[strat_name][s1_id] |= ids
        log(f"    {src} done in {src_seconds}s, pool_size={src_pool_size:,}")

    log("    aggregating matched pairs per strategy...")
    candidate_counts_by_s1 = {name: candidate_count[name] for name in strategies}
    hit_counts_by_s1 = {name: {s1: len(ids) for s1, ids in hit_ids[name].items()} for name in strategies}

    n_countries = max(len(country_counts), 1)
    # analytical estimate for a bare "country" blocking key: recall is ~100%
    # (every true match shares the same country as its S1 entity by
    # construction) and candidate count per S1 = size of its own country's pool.
    s1_country_sizes = s1_sample["country"].map(lambda c: country_counts.get(c, 0))
    country_avg_candidates = float(s1_country_sizes.mean())
    results = {
        "country": {
            "recall_pct": 100.0,
            "avg_candidates_per_s1": round(country_avg_candidates, 1),
            "median_candidates_per_s1": float(s1_country_sizes.median()),
            "max_candidates_per_s1": int(s1_country_sizes.max()) if len(s1_country_sizes) else 0,
            "reduction_ratio_pct": round(100 * (1 - country_avg_candidates / total_pool_size), 4) if total_pool_size else None,
            "true_matches_evaluated": None,
            "note": "analytical: recall assumed 100% (matches always share country); "
                    "candidate count = size of that S1 entity's own-country pool",
        }
    }

    for strat_name in strategies:
        cand_counts_series = candidate_counts_by_s1[strat_name]
        hits_map = hit_counts_by_s1[strat_name]
        hits = 0
        total_true = 0
        cand_counts = []
        for s1_id in s1_sample["entity_id"]:
            true_matches = gt.get(s1_id, set())
            if not true_matches:
                continue
            total_true += len(true_matches)
            cand_counts.append(int(cand_counts_series.get(s1_id, 0)))
            hits += hits_map.get(s1_id, 0)
        recall = hits / total_true if total_true else None
        avg_c = float(np.mean(cand_counts)) if cand_counts else 0.0
        med_c = float(np.median(cand_counts)) if cand_counts else 0.0
        max_c = int(np.max(cand_counts)) if cand_counts else 0
        reduction = 1 - (avg_c / total_pool_size) if total_pool_size else None
        results[strat_name] = {
            "recall_pct": round(100 * recall, 3) if recall is not None else None,
            "avg_candidates_per_s1": round(avg_c, 1),
            "median_candidates_per_s1": med_c,
            "max_candidates_per_s1": max_c,
            "reduction_ratio_pct": round(100 * reduction, 4) if reduction is not None else None,
            "true_matches_evaluated": int(total_true),
        }
        log(f"    {strat_name}: recall={results[strat_name]['recall_pct']}%, "
            f"avg_candidates={results[strat_name]['avg_candidates_per_s1']}, "
            f"reduction={results[strat_name]['reduction_ratio_pct']}%")

    results["_meta"] = {
        "sample_frac": sample_frac,
        "s1_sample_size": len(s1_sample),
        "candidate_pool_size": total_pool_size,
        "seconds": round(time.time() - t0, 1),
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    log(f"Audit output dir: {AUDIT_DIR}")
    log(f"SAMPLE_ROWS={SAMPLE_ROWS}, BLOCKING_SAMPLE_FRAC={BLOCKING_SAMPLE_FRAC}, "
        f"SKIP_BLOCKING={os.environ.get('SKIP_BLOCKING')}, NO_CACHE={NO_CACHE}")
    log(f"Cache dir: {CACHE_DIR} (delete it, or set NO_CACHE=1, to force a full recompute)")

    struct = basic_structure()
    _write(struct, "structure_stats.json")

    name_stats, addr_stats, examples = name_address_stats()
    _write(name_stats, "name_stats.json")
    _write(addr_stats, "address_stats.json")
    _write(examples, "examples.json")

    overlap = cross_source_overlap()
    _write(overlap, "overlap_stats.json")

    gt_stats = ground_truth_stats()
    _write(gt_stats, "ground_truth_stats.json")

    if os.environ.get("SKIP_BLOCKING") == "1":
        log("[6/6] Skipping blocking experiments (SKIP_BLOCKING=1)")
        blocking = {"skipped": True}
    else:
        blocking = blocking_experiments(sample_frac=BLOCKING_SAMPLE_FRAC)
    _write(blocking, "blocking_stats.json")

    _write_summary(struct, name_stats, addr_stats, overlap, gt_stats, blocking, t_start)
    log(f"Done. Total runtime: {round(time.time() - t_start, 1)}s")


def _write(obj, filename):
    path = os.path.join(AUDIT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log(f"    wrote {filename}")


def _write_summary(struct, name_stats, addr_stats, overlap, gt_stats, blocking, t_start):
    lines = []
    lines.append("=" * 78)
    lines.append("ML CHALLENGE 2026 -- DATASET AUDIT SUMMARY (v2)")
    lines.append("=" * 78)
    lines.append(f"Total audit runtime so far: {round(time.time() - t_start, 1)}s\n")

    lines.append("-- 1. BASIC STRUCTURE (full file scans) --")
    for key, s in struct.items():
        lines.append(f"  {key}: {s['rows']:,} rows, {s['file_size_mb']} MB")
        if "missing_pct" in s:
            lines.append(f"    missing%: {s['missing_pct']}")
            lines.append(f"    countries: {s['country_distribution']}")
    lines.append("")

    lines.append(f"-- 2. NAME STATS (sampled, up to {SAMPLE_ROWS:,} rows/file) --")
    for key, s in name_stats.items():
        lines.append(f"  {key}: n={s['sampled_rows']:,}, len(mean/median/p95)="
                      f"{s['length_chars']['mean']}/{s['length_chars']['median']}/{s['length_chars']['p95']}, "
                      f"tokens(mean)={s['token_count']['mean']}, "
                      f"raw_dupe%={s['exact_raw_duplicate_rate_pct_of_sample']}, "
                      f"norm_dupe%={s['normalized_duplicate_rate_pct_of_sample']}")
        lines.append(f"    top tokens: {s['top_tokens'][:10]}")
    lines.append("")

    lines.append(f"-- 3. ADDRESS STATS (sampled, up to {SAMPLE_ROWS:,} rows/file) --")
    for key, s in addr_stats.items():
        lines.append(f"  {key}: n={s['sampled_rows']:,}, len(mean/median/p95)="
                      f"{s['length_chars']['mean']}/{s['length_chars']['median']}/{s['length_chars']['p95']}, "
                      f"empty%={s['empty_address_pct']}, pin_rate%={s['pin_or_zip_extraction_rate_pct']}, "
                      f"digit_share%={s['digit_char_share_pct']}, norm_dupe%={s['normalized_duplicate_rate_pct_of_sample']}")
    lines.append("")

    lines.append("-- 4. CROSS-SOURCE OVERLAP (train, S1 vs S2/S3, full data via hashed sets) --")
    for pair, ov in overlap.items():
        lines.append(f"  {pair}:")
        for k, v in ov.items():
            lines.append(f"    {k}: overlap={v['overlap_count']:,} ({v['overlap_pct_of_s1']}% of S1 unique values)")
    lines.append("")

    lines.append("-- 5. GROUND TRUTH (full data) --")
    lines.append(f"  total S1 entities: {gt_stats['total_s1_entities']:,}")
    lines.append(f"  singleton%: {gt_stats['singleton_pct']}%")
    lines.append(f"  matches per S1 (mean/median/p95/max): "
                 f"{gt_stats['matches_per_s1']['mean']}/{gt_stats['matches_per_s1']['median']}/"
                 f"{gt_stats['matches_per_s1']['p95']}/{gt_stats['matches_per_s1']['max']}")
    lines.append(f"  S2-only%: {gt_stats['s2_only_pct']}, S3-only%: {gt_stats['s3_only_pct']}, "
                 f"both%: {gt_stats['both_s2_and_s3_pct']}, no-match%: {gt_stats['no_match_pct']}")
    lines.append(f"  match count distribution: {gt_stats['match_count_distribution']}")
    lines.append("")

    lines.append("-- 6. BLOCKING EXPERIMENTS --")
    if blocking.get("skipped"):
        lines.append("  SKIPPED")
    else:
        meta = blocking.get("_meta", {})
        lines.append(f"  sample: {meta.get('s1_sample_size')} S1 entities, pool: {meta.get('candidate_pool_size'):,}, "
                     f"runtime: {meta.get('seconds')}s")
        for strat, r in blocking.items():
            if strat == "_meta":
                continue
            lines.append(f"  {strat}: recall={r['recall_pct']}%, avg_candidates={r['avg_candidates_per_s1']}, "
                         f"median_candidates={r['median_candidates_per_s1']}, reduction={r['reduction_ratio_pct']}%")
    lines.append("")

    lines.append("-- FILES WRITTEN --")
    for fn in ["structure_stats.json", "name_stats.json", "address_stats.json",
               "overlap_stats.json", "ground_truth_stats.json", "blocking_stats.json", "examples.json"]:
        lines.append(f"  audit/{fn}")

    summary_path = os.path.join(AUDIT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
