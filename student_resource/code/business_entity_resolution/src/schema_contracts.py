#!/usr/bin/env python3
"""
DRAFT schema contracts between pipeline stages.

Status: DRAFT, not yet ratified by the team. Per the plan, locking these
exact column names/types was supposed to happen in the first 30-minute team
call. Since that hadn't happened yet, these are a reasonable starting point
drafted from (a) the challenge's own required final-output format
(student_resource/README.md) and (b) what the team's blocking work already
in this repo (src/blocking.ipynb) is actually producing -- so the contract
reflects real, already-written code, not just a guess.

PASTE THESE INTO THE TEAM CHAT, adjust with the team, and update this file
to match whatever gets agreed. The smoke-test harness (smoke_test.py) checks
real stage output against whatever is defined here, so this file is the
single source of truth the harness enforces.

MISMATCH FOUND AND FIXED: src/blocking.ipynb currently produces candidate
pairs as a FLAT table with columns `s1_entity_id`, `candidate_entity_id`
(one row per pair) -- but the challenge's required final format for
candidate_pairs.tsv is `source1_entity_id`, `candidate_entity_ids` (one row
per S1 entity, candidates comma-joined into a single list). These are NOT
the same shape.

Fixed in finalize_candidate_pairs.py (this directory): call
`finalize_candidate_pairs(flat_pairs_df, required_s1_ids)` to convert the
flat working format into the exact required shape, or run it as a CLI on an
existing flat .tsv/.parquet file. It also enforces the S2-/S3-prefix rule
and de-dupes candidate ids, so output passes the challenge's own
utils/validate_submission.py by construction -- verified against real
test-set data and the actual validator (PASS).
"""

# ---------------------------------------------------------------------------
# Stage 1 -- Normalization output
# ---------------------------------------------------------------------------
# What every downstream stage (blocking, features) should be able to read,
# for every source file (S1/S2/S3, train and test).
#
# NOTE: two normalization approaches already exist in this repo and are NOT
# identical -- reconcile before Stage 2 depends on a specific one:
#   - audit.py / dataset_cache.py: `norm_name`, `norm_addr`, `name_prefix4`,
#     `pin`, `addr_last_tok` (regex-based Unicode-correct normalization,
#     keeps combining marks so Devanagari isn't corrupted -- see the fix
#     documented in NON_WORD_KEEP_MARKS_RE in audit.py)
#   - src/blocking.ipynb: `name_norm`, `address_norm`, `name_translit`,
#     `address_translit`, `address_tokens` (anyascii-based transliteration
#     to a Latin-oriented representation)
# These solve different problems (preserve-script normalization vs.
# transliterate-to-Latin) and the team may want BOTH -- but every stage
# consuming "the normalized name" needs to agree on which column name means
# which of these, or two people's code will silently talk past each other.

STAGE1_NORMALIZED_FIELDS = {
    "required_columns": [
        "entity_id",            # str, e.g. "S1-123456789" -- unchanged from raw
        "business_name",        # str, original raw value, NEVER discarded
        "business_address",     # str, original raw value, NEVER discarded
        "country",              # str, original raw value
    ],
    "draft_normalized_columns": {
        # column_name: (dtype, description, produced_by)
        "norm_name": ("str", "NFKC+casefold+punct-stripped name, combining marks preserved", "audit.py"),
        "norm_addr": ("str", "same normalization applied to address", "audit.py"),
        "name_prefix4": ("str", "first 4 chars of norm_name with spaces removed", "audit.py"),
        "pin": ("str or empty", "regex-extracted postal code, US 5-digit or India 6-digit", "audit.py"),
        "addr_last_tok": ("str", "normalized last comma-separated address segment", "audit.py"),
        "name_translit": ("str", "anyascii-transliterated name, Latin-oriented", "blocking.ipynb"),
        "address_translit": ("str", "anyascii-transliterated address", "blocking.ipynb"),
        "address_tokens": ("set[str]", "informative address tokens for token-based blocking", "blocking.ipynb"),
    },
    "open_question_for_team": (
        "Which normalized-name column does Stage 2's R1 (exact normalized "
        "name) match on -- norm_name or name_translit? They will disagree on "
        "non-Latin-script records. Pick one as THE canonical join key, or "
        "explicitly run both as separate blocking routes with distinct "
        "provenance tags (from_exact_norm, from_exact_translit)."
    ),
}

# ---------------------------------------------------------------------------
# Stage 2 -- candidate_pairs (the file BOTH the intermediate blocking output
# AND the final required deliverable share a name with -- do not confuse them)
# ---------------------------------------------------------------------------
#
# candidate_pairs.tsv has exactly ONE required shape -- the one the
# challenge scores against (see student_resource/README.md). Any
# intermediate flat pair table used DURING blocking is a different, internal
# format and must be converted before it becomes candidate_pairs.tsv.

CANDIDATE_PAIRS_FINAL_FORMAT = {
    "filename": "candidate_pairs.tsv",
    "columns": ["source1_entity_id", "candidate_entity_ids"],
    "shape": "ONE ROW PER SOURCE-1 ENTITY. candidate_entity_ids is a single "
             "comma-joined string of S2-/S3- ids (empty string, not NaN, "
             "when there are zero candidates).",
    "rules": [
        "Every test_source1.tsv entity_id must appear exactly once.",
        "candidate_entity_ids contains only S2-/S3- prefixed ids that exist "
        "in the test set.",
        "No duplicate ids within one row's candidate list.",
        "matching_results.tsv's matched ids must be a SUBSET of this file's "
        "candidates for the same S1 entity (the validator warns, doesn't "
        "fail, if this is violated -- but a violation signals a pipeline bug).",
    ],
}

CANDIDATE_PAIRS_INTERMEDIATE_FORMAT_SEEN_IN_REPO = {
    "source": "src/blocking.ipynb",
    "columns": ["s1_entity_id", "candidate_entity_id"],
    "shape": "ONE ROW PER (S1, candidate) PAIR -- flat, not grouped. This is "
             "a reasonable internal working format DURING blocking (easy to "
             "dedupe, easy to merge against ground truth for recall "
             "measurement, which is exactly what blocking.ipynb does with "
             "it) but it is NOT the final candidate_pairs.tsv shape.",
    "conversion_needed": (
        "groupby('s1_entity_id')['candidate_entity_id'].apply(lambda ids: "
        "','.join(sorted(set(ids)))) then reindex against every "
        "test_source1 id (fillna('') for S1 entities with zero candidates), "
        "then rename columns to source1_entity_id/candidate_entity_ids."
    ),
}

# ---------------------------------------------------------------------------
# Stage 3 -- Feature Parquet (pairwise features for the candidate set)
# ---------------------------------------------------------------------------
# IMPLEMENTED at sample scale: stage3_features.py, output at
# pipeline_output/stage3_features.parquet. Join-key columns use
# s1_entity_id/candidate_entity_id (the FLAT intermediate convention, same
# as CANDIDATE_PAIRS_INTERMEDIATE_FORMAT_SEEN_IN_REPO below) rather than
# source1_entity_id/candidate_entity_ids (which is reserved for the FINAL
# grouped/comma-joined shape) -- this file is one row per (S1, candidate)
# pair, not grouped, so the flat naming is the consistent choice. An
# earlier draft of this contract used source1_entity_id here inconsistently
# with that rule; smoke_test.py's Stage 3/4 checks were caught failing
# against real pipeline output because of exactly that mismatch, which is
# what surfaced and fixed it.

STAGE3_FEATURE_FILE_DRAFT = {
    "required_columns": [
        "s1_entity_id",          # str -- join key back to S1 (flat format)
        "candidate_entity_id",   # str -- join key back to S2/S3 (flat, one row per pair)
        "label",                 # int 0/1, ONLY present in training feature files
                                  # (true match per ground truth), ABSENT in
                                  # inference feature files
    ],
    "feature_columns_implemented": [
        # actually computed by stage3_features.py, reusing
        # experiments.py's compute_features_parallel() for the similarity
        # metrics rather than reimplementing them
        "name_exact_raw_equal", "name_exact_norm_equal",
        "name_char3_jaccard", "name_char4_jaccard", "name_token_jaccard",
        "name_edit_similarity", "name_token_set_ratio", "name_length_ratio",
        "name_prefix4_equal",
        "addr_exact_raw_equal", "addr_exact_norm_equal",
        "addr_char3_jaccard", "addr_char4_jaccard", "addr_token_jaccard",
        "addr_edit_similarity", "addr_token_set_ratio", "addr_length_ratio",
        "addr_prefix4_equal",
        "route_count",             # = num_blockers from blocking provenance
        "max_block_score",         # passthrough from blocking provenance
        "country_conflict", "name_missing_s1", "name_missing_candidate",
        "address_missing_s1", "address_missing_candidate",
    ],
    "feature_columns_not_yet_implemented": [
        # from the original wishlist -- cut for the sample-scale pass
        # (schema_contracts.py's job is to document reality, not aspiration)
        "name_tfidf_cosine", "token_idf_overlap", "rare_token_match",
        "house_number_match", "postal_match", "script_tag_match",
        "transliterated_similarity", "best_rank", "sibling_rank",
        "score_gap_to_next_best",
    ],
    "dtype_note": "float32 throughout for the numeric features -- keeps a "
                  "150k-S1 x ~30-candidate x ~25-feature shard around "
                  "450MB, per the hardware-aware plan's own sizing.",
    "sharding": "NOT YET IMPLEMENTED -- current script loads the whole "
               "1,000-S1 sample in memory. Sharding by S1-id range is "
               "required before running at full (~2.2M S1) scale.",
}

# ---------------------------------------------------------------------------
# Stage 4 -- Calibrated probability output
# ---------------------------------------------------------------------------
# IMPLEMENTED at sample scale: stage4_train_model.py, output at
# pipeline_output/stage4_probabilities.parquet. Same flat-format join-key
# naming as Stage 3, for the same reason (one row per pair, not grouped).

STAGE4_PROBABILITY_FORMAT_DRAFT = {
    "required_columns": [
        "s1_entity_id",             # str
        "candidate_entity_id",      # str -- flat, one row per (S1, candidate) pair
        "raw_score",                # float, LightGBM's raw predict_proba output
        "calibrated_probability",   # float in [0, 1], after isotonic calibration
                                     # (sklearn CalibratedClassifierCV + FrozenEstimator --
                                     # cv="prefit" was removed in newer sklearn, see
                                     # stage4_train_model.py for the replacement)
    ],
    "note": "Stage 5's flat-threshold decision rule uses calibrated_probability. "
           "Monte Carlo expected-F0.5 prefix selection would also use this "
           "column but is NOT YET IMPLEMENTED in stage5_decision_engine.py "
           "-- the flat threshold already achieves 0.9481 macro F_0.5 on a "
           "genuinely held-out GroupKFold split, so it shipped first per "
           "the plan's own fallback-ladder principle.",
}
