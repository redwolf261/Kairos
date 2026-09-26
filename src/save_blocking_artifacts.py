from pathlib import Path
from datetime import datetime
import json
import pandas as pd


def save_blocking_artifacts(
    candidates,
    comparison,
    output_dir,
    parameters,
    source_name="experiment2",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidates.copy()

    required = {
        "s1_entity_id",
        "candidate_entity_id",
    }

    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    candidates = candidates.drop_duplicates(
        subset=[
            "s1_entity_id",
            "candidate_entity_id",
        ]
    )

    candidate_file = (
        output_dir
        / f"{source_name}_candidate_pairs.tsv"
    )

    comparison_file = (
        output_dir
        / f"{source_name}_blocking_comparison.tsv"
    )

    manifest_file = (
        output_dir
        / f"{source_name}_blocking_manifest.json"
    )

    candidates.to_csv(
        candidate_file,
        sep="\t",
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        sep="\t",
        index=False,
    )

    manifest = {
        "experiment": source_name,
        "created_at_utc": datetime.utcnow().isoformat(),
        "candidate_rows": int(len(candidates)),
        "candidate_columns": list(candidates.columns),
        "candidate_file": candidate_file.name,
        "comparison_file": comparison_file.name,
        "parameters": parameters,
    }

    with manifest_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            default=str,
        )

    print(f"Saved candidates: {candidate_file}")
    print(f"Saved comparison: {comparison_file}")
    print(f"Saved manifest: {manifest_file}")

    return manifest