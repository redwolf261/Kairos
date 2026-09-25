# Kairos — Business Entity Resolution

An ML pipeline for the **ML Challenge 2026: Business Entity Resolution** —
matching business records across three noisy, independent data sources
(~12M records total) using blocking + similarity features, under a
precision-heavy F₀.₅ scoring metric. See
[`student_resource/README.md`](student_resource/README.md) for the full
problem statement.

This repo currently holds the **data audit, caching, and similarity-experiment
tooling** used to understand the dataset before building the final blocking +
matching model. The raw dataset itself is *not* committed here (see
[Getting the data](#getting-the-data) below).

## Repo layout

```
student_resource/
├── README.md                     # original challenge problem statement
├── Documentation_template.md     # methodology write-up template (for final submission)
├── utils/
│   └── validate_submission.py    # official output-format validator
├── code/business_entity_resolution/src/
│   ├── audit.py                  # dataset audit: structure, name/address stats,
│   │                              # cross-source overlap, ground-truth stats,
│   │                              # single-strategy blocking recall
│   ├── dataset_cache.py          # shared normalized-dataset cache (Parquet),
│   │                              # used by every other script instead of
│   │                              # re-parsing/re-normalizing raw TSVs each time
│   ├── experiments.py            # Experiments 1–5: true-match similarity
│   │                              # distributions, true-vs-false pair
│   │                              # comparison, blocking-miss analysis,
│   │                              # transliteration/script analysis, and a
│   │                              # union-blocking recall sweep
│   └── export_10k_sample.py      # exports a reproducible 10k-S1-entity
│                                   # experimental sample
├── audit/                        # JSON/TXT results from audit.py + experiments.py
├── experiment_10k/               # reproducible 10k-entity sample (seed=42)
│   ├── s1_sample.tsv
│   ├── true_matches.tsv
│   └── metadata.json
└── dataset/                      # NOT committed — see below
```

## Getting the data

The dataset (`train_source1/2/3.tsv`, `train_ground_truth.tsv`,
`test_source1/2/3.tsv`, ~2.4GB total) is provided by the challenge and is
**gitignored** — it's not pushed to this repo. To reproduce anything here:

1. Place the challenge's `dataset/` folder at `student_resource/dataset/`
   (i.e. `student_resource/dataset/train/*.tsv` and
   `student_resource/dataset/test/*.tsv`).
2. Run the scripts below from the `student_resource/` directory.

## Setup

```bash
pip install pandas numpy pyarrow rapidfuzz regex
```

(`pyarrow` backs the Parquet cache; `rapidfuzz` and `regex` are used for
similarity features and Unicode-correct text normalization respectively —
both MIT-licensed, no external API calls.)

## Running the pipeline

All scripts are cached and parallelized across all available CPU cores — a
cold run processes the full dataset in a few minutes, and a re-run with
unchanged inputs completes in under a second (see
[Caching](#caching--performance) below).

```bash
cd student_resource

# 1. Build the shared normalized dataset cache (name/address normalization,
#    PIN extraction, etc. computed once per source file)
python code/business_entity_resolution/src/dataset_cache.py

# 2. Run the dataset audit (structure, name/address stats, cross-source
#    overlap, ground-truth stats, single-strategy blocking recall)
python code/business_entity_resolution/src/audit.py

# 3. Run Experiments 1-5 (similarity distributions, blocking-miss analysis,
#    transliteration analysis, union-blocking recall sweep)
python code/business_entity_resolution/src/experiments.py

# 4. Export a reproducible 10k-S1-entity sample for further experimentation
python code/business_entity_resolution/src/export_10k_sample.py
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SAMPLE_ROWS` | `500000` | Rows sampled per file for detailed name/address stats |
| `BLOCKING_SAMPLE_FRAC` | `0.01` | Fraction of S1 entities used in `audit.py`'s blocking test |
| `EXPERIMENT_SAMPLE_FRAC` | `0.02` | Fraction of S1 entities used in Experiments 1–4 |
| `EXP5_SAMPLE_FRAC` | `0.01` | Fraction of S1 entities used in Experiment 5 (the most expensive) |
| `AUDIT_WORKERS` | CPU count | Number of parallel worker processes |
| `NO_CACHE=1` | off | Bypass all caches and recompute from scratch |

## Caching & performance

- **Shared dataset cache** (`dataset_cache.py`): each source `.tsv` is
  normalized once (name/address normalization, PIN extraction, etc.) and
  cached as Parquet under `audit/cache/dataset/`, fingerprinted on file
  size + mtime. Every other script reads from this cache instead of
  re-parsing the raw TSV.
- **Step-level cache** (`audit.py`'s `load_cached`/`save_cache`): every
  audit/experiment step's result is cached under `audit/cache/`, keyed on
  the fingerprints of the files it reads plus its own parameters. Editing or
  replacing a source `.tsv` automatically invalidates only the affected
  steps.
- **Parallelization**: independent per-file/per-strategy work (e.g. scanning
  all 7 source files, or the two candidate pool files) is dispatched across
  a `ProcessPoolExecutor` sized to the machine's core count.

## Key findings so far

Full detail in `student_resource/audit/summary.txt` and
`student_resource/audit/experiments_summary.txt`. Headlines:

- **~64.8%** of Source-1 normalized names appear (exactly, after
  normalization) somewhere in Source 2; **~65%** for Source 3 — but only
  **42.8%** of *true matches* have exactly-equal normalized names, meaning
  most real matches contain enough noise that exact-name blocking alone
  misses the majority of them.
- **88.1%** of true matches have strong (≥0.6 Jaccard) character-3-gram
  overlap in name *or* address — a combined similarity signal recovers the
  large majority of true matches even when exact-name blocking doesn't.
- Of the ~57% of true matches missed by `country + normalized_name`
  blocking, roughly **65%** are recoverable noise (abbreviations, typos,
  word-order differences) and only **~15%** look like genuinely hard cases.
- **~17.5%** of India true matches with unequal names are clean
  Latin↔Devanagari transliterations (e.g. "Jay Constructions Private
  Limited" ↔ "जय कंस्ट्रक्शंस प्राइवेट लिमिटेड") — recoverable with
  script-aware normalization, not a source of irreducible noise.
- Naive prefix-based union blocking plateaus around **85–86% recall** at
  ~13,000 candidates/S1 — too loose to use directly; a similarity-based
  (n-gram/token) retrieval layer is the likely next step rather than wider
  prefix unions.

## Constraints (per challenge rules)

- No external data lookups, APIs, or services — all normalization and
  similarity computation here is local/offline (Unicode codepoint ranges,
  regex, rapidfuzz string similarity).
- Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.

## Status

Data audit and candidate-generation research phase. Blocking + pairwise
matching model not yet built.
