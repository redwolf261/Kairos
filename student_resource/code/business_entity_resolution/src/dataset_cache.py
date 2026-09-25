#!/usr/bin/env python3
"""
ML Challenge 2026 -- Shared Normalized Dataset Cache

Every experiment script (the audit, the future similarity-distribution
experiments, blocking-strategy sweeps, the eventual matching model) needs the
same base thing: each source file's records with normalization ALREADY
applied (norm_name, norm_addr, name_prefix4, pin, addr_last_tok). Right now
every script/step that wants that has to re-read the raw .tsv and re-run
regex/NFKC normalization itself -- correct, but wasteful when ten different
experiments all want the same normalized columns.

This module builds that normalized dataset ONCE per source file and caches it
as Parquet (columnar, fast to re-read, far cheaper than re-parsing 200-500MB
of raw TSV + re-running normalization every time). Every future script should
import `load_normalized(file_key)` from here instead of reading the raw .tsv
directly.

Caching behaviour (same fingerprint scheme as audit.py's cache):
  - Keyed on (file size, mtime) of the source .tsv -- NOT its content hash,
    so validating the cache never requires rereading 12M rows.
  - A stale/missing Parquet cache is rebuilt transparently, from a streaming
    chunked read (bounded memory even for the 5M+ row files).
  - Editing or replacing a .tsv under the same name changes its fingerprint
    and transparently invalidates only that file's cached Parquet.

Usage:
    from dataset_cache import load_normalized, load_normalized_lazy

    # full DataFrame in memory (fine for train_source1 / test_source1, ~2M rows)
    df = load_normalized("train_source1")

    # for the big S2/S3 files (~5M rows), prefer a filtered/chunked read:
    df = load_normalized("train_source2", entity_ids={"S2-123", "S2-456"})

    # or iterate in chunks without ever holding the whole file in memory:
    for chunk in load_normalized_lazy("train_source2"):
        ...

Columns in the normalized Parquet (source files only; ground truth is
handled separately since it has a different schema):
    entity_id, business_name, business_address, country,
    norm_name, norm_addr, name_prefix4, pin, addr_last_tok

Run this file directly to (re)build the cache for every source file:
    python code/business_entity_resolution/src/dataset_cache.py
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402  (reuses FILES, normalization fns, cache dir, iter_chunks)

DATASET_CACHE_DIR = os.path.join(audit.CACHE_DIR, "dataset")
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

NORMALIZED_COLUMNS = [
    "entity_id", "business_name", "business_address", "country",
    "norm_name", "norm_addr", "name_prefix4", "pin", "addr_last_tok",
]


def _parquet_path(file_key, fingerprint):
    safe_fp = fingerprint.replace(":", "_").replace(os.sep, "_")
    return os.path.join(DATASET_CACHE_DIR, f"{file_key}.{safe_fp}.parquet")


def _find_valid_cache(file_key):
    """Return the path to a valid (fingerprint-matching) cached Parquet for
    this file, or None. Also removes stale Parquet files for this file_key
    (old fingerprint) so the cache dir doesn't accumulate garbage copies of
    a multi-hundred-MB file."""
    fingerprint = audit._file_fingerprint(audit.FILES[file_key])
    valid_path = _parquet_path(file_key, fingerprint)
    prefix = f"{file_key}."
    for fname in os.listdir(DATASET_CACHE_DIR):
        full = os.path.join(DATASET_CACHE_DIR, fname)
        if fname.startswith(prefix) and full != valid_path:
            try:
                os.remove(full)
                audit.log(f"    [dataset_cache] removed stale {fname}")
            except OSError:
                pass
    return valid_path if os.path.isfile(valid_path) else None


def _build_normalized_parquet(file_key):
    """Stream the raw .tsv, add normalized columns, write Parquet. Uses
    bounded-memory chunked reads and writes incrementally via a
    pyarrow.parquet.ParquetWriter so even the 5M+ row files never require
    holding the full normalized DataFrame in memory at once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = audit.FILES[file_key]
    fingerprint = audit._file_fingerprint(path)
    out_path = _parquet_path(file_key, fingerprint)
    tmp_path = out_path + ".tmp"

    t0 = time.time()
    n_rows = 0
    writer = None
    try:
        for chunk in audit.iter_chunks(
            path, usecols=["entity_id", "business_name", "business_address", "country"]
        ):
            chunk["norm_name"] = audit.normalize_name_series(chunk["business_name"])
            chunk["norm_addr"] = audit.normalize_series(chunk["business_address"])
            chunk["name_prefix4"] = audit.name_prefix4_series(chunk["norm_name"])
            chunk["pin"] = audit.extract_pin_series(chunk["business_address"]).fillna("")
            chunk["addr_last_tok"] = audit.addr_last_token_series(chunk["business_address"])
            chunk = chunk[NORMALIZED_COLUMNS]

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
            writer.write_table(table)
            n_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, out_path)
    audit.log(f"    [dataset_cache] built {file_key}: {n_rows:,} rows -> "
              f"{out_path} in {round(time.time()-t0,1)}s "
              f"({round(os.path.getsize(out_path)/1e6,1)} MB)")
    return out_path


def ensure_cached(file_key):
    """Return the path to a valid cached Parquet for this source file,
    building it first if missing/stale. Safe to call repeatedly -- a valid
    cache is a no-op fingerprint check, not a re-read of the .tsv."""
    valid = _find_valid_cache(file_key)
    if valid is not None:
        return valid
    return _build_normalized_parquet(file_key)


def load_normalized(file_key, entity_ids=None, columns=None):
    """Load a source file's normalized data as a single DataFrame.

    entity_ids: optional iterable/set of entity_id values to filter to
        (pushed down at the Parquet row-group level where possible) --
        use this for the big S2/S3 files instead of loading all 5M rows
        when you only need a few thousand specific records.
    columns: optional list of columns to read (defaults to all normalized
        columns) -- Parquet only reads the columns you ask for, so skip
        business_name/business_address if you only need the normalized
        forms, for a faster/lighter read.
    """
    import pyarrow.parquet as pq

    path = ensure_cached(file_key)
    filters = None
    if entity_ids is not None:
        filters = [("entity_id", "in", list(entity_ids))]
    table = pq.read_table(path, columns=columns, filters=filters)
    return table.to_pandas()


def load_normalized_lazy(file_key, columns=None, batch_size=200_000):
    """Yield the normalized data in batches (as DataFrames) without ever
    holding the full file in memory -- use this for full scans over the big
    S2/S3/test files (e.g. blocking experiments, similarity sweeps)."""
    import pyarrow.parquet as pq

    path = ensure_cached(file_key)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def build_all(file_keys=None):
    """(Re)build the normalized Parquet cache for every source file (or the
    given subset). Ground truth is skipped -- it has a different schema and
    is cheap enough to read raw each time (handled directly in audit.py)."""
    keys = file_keys or [k for k in audit.FILES if "ground_truth" not in k]
    audit.log(f"Building shared normalized dataset cache for: {keys}")
    t0 = time.time()

    # independent per-file work -- parallelize across all available cores,
    # same executor sizing as the rest of the audit pipeline
    with audit.make_executor() as ex:
        futures = {ex.submit(ensure_cached, k): k for k in keys}
        for fut in audit.as_completed(futures):
            key = futures[fut]
            path = fut.result()
            audit.log(f"    {key}: ready at {path}")

    audit.log(f"Dataset cache build/verify done in {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    build_all()
