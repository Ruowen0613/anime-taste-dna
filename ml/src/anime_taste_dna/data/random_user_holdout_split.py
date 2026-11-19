from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anime_taste_dna.data.user_holdout_split import assign_user_groups

OUTPUT_NAMES = (
    "train",
    "validation_history",
    "validation_target",
    "test_history",
    "test_target",
)


@dataclass(frozen=True)
class RandomUserHoldoutConfig:
    ratings_path: Path
    preprocessing_report_path: Path
    output_directory: Path
    working_directory: Path
    validation_user_fraction: float
    test_user_fraction: float
    target_fraction: float
    minimum_target_interactions: int
    user_buckets: int
    batch_size: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> RandomUserHoldoutConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        processed_directory = resolve(raw["paths"]["processed_directory"])
        return cls(
            ratings_path=processed_directory / "ratings.parquet",
            preprocessing_report_path=processed_directory / "preprocessing_report.json",
            output_directory=resolve(raw["paths"]["output_directory"]),
            working_directory=resolve(raw["paths"]["working_directory"]),
            validation_user_fraction=float(raw["split"]["validation_user_fraction"]),
            test_user_fraction=float(raw["split"]["test_user_fraction"]),
            target_fraction=float(raw["split"]["target_fraction"]),
            minimum_target_interactions=int(raw["split"]["minimum_target_interactions"]),
            user_buckets=int(raw["split"]["user_buckets"]),
            batch_size=int(raw["split"]["batch_size"]),
            seed=int(raw["split"]["seed"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def random_target_positions(
    interaction_count: int,
    target_fraction: float,
    minimum_target_interactions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if interaction_count < 2:
        raise ValueError("Held-out users need at least two interactions")
    target_count = max(
        minimum_target_interactions,
        int(np.floor(interaction_count * target_fraction)),
    )
    target_count = min(target_count, interaction_count - 1)
    permutation = np.random.default_rng(seed).permutation(interaction_count)
    return permutation[target_count:], permutation[:target_count]


def _partition(config: RandomUserHoldoutConfig, bucket_directory: Path) -> int:
    source = pq.ParquetFile(config.ratings_path)
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    try:
        for batch in source.iter_batches(
            columns=["user_index", "item_index", "mal_id", "rating"],
            batch_size=config.batch_size,
        ):
            frame = batch.to_pandas()
            bucket_ids = frame["user_index"].to_numpy() % config.user_buckets
            for bucket_id in np.unique(bucket_ids):
                selected = frame.loc[bucket_ids == bucket_id]
                table = pa.Table.from_pandas(selected, preserve_index=False)
                bucket_number = int(bucket_id)
                writer = writers.get(bucket_number)
                if writer is None:
                    writer = pq.ParquetWriter(
                        bucket_directory / f"bucket-{bucket_number:04d}.parquet",
                        table.schema,
                        compression="zstd",
                    )
                    writers[bucket_number] = writer
                writer.write_table(table)
                rows += len(selected)
    finally:
        for writer in writers.values():
            writer.close()
    return rows


def _write_rows(
    writers: dict[str, pq.ParquetWriter],
    output_directory: Path,
    name: str,
    frame: pd.DataFrame,
) -> int:
    if frame.empty:
        return 0
    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = writers.get(name)
    if writer is None:
        writer = pq.ParquetWriter(
            output_directory / f"{name}.parquet",
            table.schema,
            compression="zstd",
        )
        writers[name] = writer
    writer.write_table(table)
    return len(frame)


def create_random_user_holdout_split(config: RandomUserHoldoutConfig) -> dict[str, Any]:
    if not config.ratings_path.is_file() or not config.preprocessing_report_path.is_file():
        raise FileNotFoundError("Processed ratings and preprocessing report must exist")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_directory}")
    if config.working_directory.exists():
        raise FileExistsError(f"Working directory already exists: {config.working_directory}")
    if config.validation_user_fraction + config.test_user_fraction >= 1:
        raise ValueError("Held-out user fractions must sum to less than one")

    preprocessing_report = json.loads(
        config.preprocessing_report_path.read_text(encoding="utf-8")
    )
    users = int(preprocessing_report["output"]["users"])
    items = int(preprocessing_report["output"]["items"])
    expected_rows = int(preprocessing_report["output"]["ratings"])
    if _sha256(config.ratings_path) != preprocessing_report["output"]["ratings_sha256"]:
        raise ValueError("Processed ratings fingerprint does not match preprocessing report")

    groups = assign_user_groups(
        users,
        config.validation_user_fraction,
        config.test_user_fraction,
        config.seed,
    )
    config.output_directory.mkdir(parents=True)
    bucket_directory = config.working_directory / "user-buckets"
    bucket_directory.mkdir(parents=True)
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = {name: 0 for name in OUTPUT_NAMES}
    user_sets: dict[str, set[int]] = {name: set() for name in OUTPUT_NAMES}
    try:
        partitioned_rows = _partition(config, bucket_directory)
        for bucket_path in sorted(bucket_directory.glob("bucket-*.parquet")):
            frame = pq.read_table(bucket_path).to_pandas()
            frame = frame.sort_values(["user_index", "item_index"], kind="mergesort")
            for user_index, positions in frame.groupby("user_index", sort=False).indices.items():
                user_rows = frame.iloc[np.asarray(positions, dtype=np.int64)].reset_index(drop=True)
                group = int(groups[int(user_index)])
                if group == 0:
                    row_counts["train"] += _write_rows(
                        writers, config.output_directory, "train", user_rows
                    )
                    user_sets["train"].add(int(user_index))
                    continue

                history_positions, target_positions = random_target_positions(
                    len(user_rows),
                    config.target_fraction,
                    config.minimum_target_interactions,
                    config.seed ^ int(user_index),
                )
                prefix = "validation" if group == 1 else "test"
                for suffix, selected_positions in (
                    ("history", history_positions),
                    ("target", target_positions),
                ):
                    name = f"{prefix}_{suffix}"
                    selected = user_rows.iloc[selected_positions]
                    row_counts[name] += _write_rows(
                        writers, config.output_directory, name, selected
                    )
                    user_sets[name].add(int(user_index))
    finally:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(config.working_directory, ignore_errors=True)

    if partitioned_rows != expected_rows or sum(row_counts.values()) != expected_rows:
        raise RuntimeError("Random user-holdout split row counts do not match input")
    if user_sets["validation_history"] != user_sets["validation_target"]:
        raise RuntimeError("Validation history and target users differ")
    if user_sets["test_history"] != user_sets["test_target"]:
        raise RuntimeError("Test history and target users differ")
    if (
        user_sets["train"] & user_sets["validation_history"]
        or user_sets["train"] & user_sets["test_history"]
        or user_sets["validation_history"] & user_sets["test_history"]
    ):
        raise RuntimeError("User groups overlap")

    files = {
        name: {
            "rows": row_counts[name],
            "users": len(user_sets[name]),
            "sha256": _sha256(config.output_directory / f"{name}.parquet"),
        }
        for name in OUTPUT_NAMES
    }
    report = {
        "source": {
            "ratings_sha256": preprocessing_report["output"]["ratings_sha256"],
            "ratings": expected_rows,
            "users": users,
            "items": items,
        },
        "protocol": {
            "name": "strict user holdout with deterministic random targets; timestamps ignored",
            "user_group_seed": config.seed,
            "validation_user_fraction": config.validation_user_fraction,
            "test_user_fraction": config.test_user_fraction,
            "target_fraction": config.target_fraction,
            "minimum_target_interactions": config.minimum_target_interactions,
            "global_training_users": "train group only",
        },
        "user_groups": {
            "train": len(user_sets["train"]),
            "validation": len(user_sets["validation_history"]),
            "test": len(user_sets["test_history"]),
            "pairwise_disjoint": True,
        },
        "output": {"files": files},
    }
    report_path = config.output_directory / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a random-target strict user split")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = RandomUserHoldoutConfig.from_toml(args.config)
    print(json.dumps(create_random_user_holdout_split(config), indent=2))


if __name__ == "__main__":
    main()
