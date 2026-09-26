from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42
TEST_SIZE = 0.20
NEGATIVE_TO_POSITIVE_RATIO = 3


def normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value):
    return set(normalize(value).split())


def jaccard(left, right):
    left = tokens(left)
    right = tokens(right)

    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def make_text(row, column):
    return normalize(row.get(column, ""))


def build_features(left, right, vectorizer):
    name_left = make_text(left, "business_name")
    name_right = make_text(right, "business_name")

    address_left = make_text(left, "business_address")
    address_right = make_text(right, "business_address")

    full_left = f"{name_left} {address_left}".strip()
    full_right = f"{name_right} {address_right}".strip()

    full_vectors = vectorizer.transform([full_left, full_right])
    full_cosine = cosine_similarity(
        full_vectors[0],
        full_vectors[1],
    )[0, 0]

    name_vectors = vectorizer.transform([name_left, name_right])
    name_cosine = cosine_similarity(
        name_vectors[0],
        name_vectors[1],
    )[0, 0]

    return {
        "name_exact": int(name_left == name_right and name_left != ""),
        "address_exact": int(
            address_left == address_right and address_left != ""
        ),
        "full_exact": int(
            full_left == full_right and full_left != ""
        ),
        "name_token_jaccard": jaccard(name_left, name_right),
        "address_token_jaccard": jaccard(
            address_left,
            address_right,
        ),
        "full_token_jaccard": jaccard(full_left, full_right),
        "name_levenshtein": ratio(name_left, name_right) / 100,
        "address_levenshtein": (
            ratio(address_left, address_right) / 100
        ),
        "full_levenshtein": ratio(full_left, full_right) / 100,
        "name_token_set_ratio": (
            token_set_ratio(name_left, name_right) / 100
        ),
        "full_token_set_ratio": (
            token_set_ratio(full_left, full_right) / 100
        ),
        "name_cosine": float(name_cosine),
        "full_cosine": float(full_cosine),
        "name_left_length": len(name_left),
        "name_right_length": len(name_right),
        "address_left_length": len(address_left),
        "address_right_length": len(address_right),
        "name_left_tokens": len(name_left.split()),
        "name_right_tokens": len(name_right.split()),
        "address_left_tokens": len(address_left.split()),
        "address_right_tokens": len(address_right.split()),
        "same_country": int(
            normalize(left.get("country_norm", ""))
            == normalize(right.get("country_norm", ""))
        ),
    }


def build_dataset(
    s1,
    s2,
    s3,
    candidate_pairs,
    ground_truth,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = pd.concat(
        [s2, s3],
        ignore_index=True,
    ).copy()

    source_records["entity_id"] = (
        source_records["entity_id"].astype(str)
    )
    s1["entity_id"] = s1["entity_id"].astype(str)

    candidate_pairs = candidate_pairs[
        ["s1_entity_id", "candidate_entity_id"]
    ].drop_duplicates()

    ground_truth = ground_truth[
        ["s1_entity_id", "true_entity_id"]
    ].drop_duplicates()

    # Add missed true pairs so every known positive is available.
    missed_pairs = ground_truth.rename(
        columns={"true_entity_id": "candidate_entity_id"}
    )

    pairs = pd.concat(
        [candidate_pairs, missed_pairs],
        ignore_index=True,
    ).drop_duplicates()

    pairs = pairs.merge(
        ground_truth.assign(is_same=1),
        left_on=[
            "s1_entity_id",
            "candidate_entity_id",
        ],
        right_on=[
            "s1_entity_id",
            "true_entity_id",
        ],
        how="left",
    )

    pairs["is_same"] = pairs["is_same"].fillna(0).astype(int)
    pairs = pairs.drop(columns=["true_entity_id"], errors="ignore")

    s1_lookup = s1.set_index("entity_id").to_dict("index")
    source_lookup = source_records.set_index(
        "entity_id"
    ).to_dict("index")

    all_text = pd.concat(
        [
            s1.get("business_name", pd.Series(dtype=str)),
            s1.get("business_address", pd.Series(dtype=str)),
            source_records.get(
                "business_name",
                pd.Series(dtype=str),
            ),
            source_records.get(
                "business_address",
                pd.Series(dtype=str),
            ),
        ]
    ).fillna("").map(normalize)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
    )
    vectorizer.fit(all_text)

    feature_rows = []

    for _, pair in pairs.iterrows():
        left = s1_lookup[pair["s1_entity_id"]]
        right = source_lookup[pair["candidate_entity_id"]]

        features = build_features(
            left,
            right,
            vectorizer,
        )

        features["s1_entity_id"] = pair["s1_entity_id"]
        features["candidate_entity_id"] = (
            pair["candidate_entity_id"]
        )
        features["is_same"] = pair["is_same"]

        feature_rows.append(features)

    dataset = pd.DataFrame(feature_rows)

    positives = dataset[dataset["is_same"] == 1]
    negatives = dataset[dataset["is_same"] == 0]

    max_negatives = len(positives) * NEGATIVE_TO_POSITIVE_RATIO

    if len(negatives) > max_negatives:
        negatives = negatives.sample(
            n=max_negatives,
            random_state=RANDOM_STATE,
        )

    dataset = pd.concat(
        [positives, negatives],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    groups = dataset["s1_entity_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            dataset,
            dataset["is_same"],
            groups=groups,
        )
    )

    train = dataset.iloc[train_indices].reset_index(drop=True)
    test = dataset.iloc[test_indices].reset_index(drop=True)

    dataset.to_csv(
        output_dir / "experiment2_model_dataset.tsv",
        sep="\t",
        index=False,
    )

    train.to_csv(
        output_dir / "experiment2_train.tsv",
        sep="\t",
        index=False,
    )

    test.to_csv(
        output_dir / "experiment2_test.tsv",
        sep="\t",
        index=False,
    )

    metadata = {
        "total_rows": len(dataset),
        "positive_rows": int(dataset["is_same"].sum()),
        "negative_rows": int((dataset["is_same"] == 0).sum()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rows": int(train["is_same"].sum()),
        "test_positive_rows": int(test["is_same"].sum()),
        "test_size": TEST_SIZE,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "split_strategy": "GroupShuffleSplit by s1_entity_id",
        "feature_columns": [
            column
            for column in dataset.columns
            if column not in [
                "s1_entity_id",
                "candidate_entity_id",
                "is_same",
            ]
        ],
    }

    with open(
        output_dir / "experiment2_dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))

    return dataset, train, test