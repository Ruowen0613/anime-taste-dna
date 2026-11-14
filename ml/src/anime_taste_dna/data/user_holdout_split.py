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

OUTPUT_NAMES = (
    "train",
    "validation_history",
    "validation_target",
    "test_history",
    "test_target",
)


@dataclass(frozen=True)
class UserHoldoutConfig:
    ratings_path: Path
    preprocessing_report_path: Path
    output_directory: Path
    working_directory: Path
    validation_user_fraction: float
    test_user_fraction: float
    target_fraction: float
    minimum_target_interactions: int
    minimum_history_interactions: int
    user_buckets: int
    batch_size: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> UserHoldoutConfig:
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
            minimum_history_interactions=int(raw["split"]["minimum_history_interactions"]),
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


def assign_user_groups(
    users: int, validation_fraction: float, test_fraction: float, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(users)
    validation_count = int(np.floor(users * validation_fraction))
    test_count = int(np.floor(users * test_fraction))
    groups = np.zeros(users, dtype=np.int8)
    groups[permutation[:validation_count]] = 1
    groups[permutation[validation_count : validation_count + test_count]] = 2
    return groups


def label_user_rows(
    frame: pd.DataFrame,
    user_groups: np.ndarray,
    target_fraction: float,
    minimum_target: int,
    minimum_history: int,
) -> pd.DataFrame:
    frame = frame.sort_values(
        ["user_index", "updated_at", "item_index"], kind="mergesort"
    ).reset_index(drop=True)
    labels = np.empty(len(frame), dtype=object)
    for user_index, positions in frame.groupby("user_index", sort=False).indices.items():
        row_indices = np.asarray(positions, dtype=np.int64)
        group = int(user_groups[int(user_index)])
        if group == 0:
            labels[row_indices] = "train"
            continue
        target_count = max(int(np.floor(len(row_indices) * target_fraction)), minimum_target)
        history_count = len(row_indices) - target_count
        if history_count < minimum_history:
            raise ValueError(
                f"User {user_index} has {history_count} history interactions; "
                f"at least {minimum_history} are required"
            )
        prefix = "validation" if group == 1 else "test"
        labels[row_indices[:history_count]] = f"{prefix}_history"
        labels[row_indices[history_count:]] = f"{prefix}_target"
    result = frame.copy()
    result["split"] = labels
    return result


def _validate(config: UserHoldoutConfig) -> dict[str, Any]:
    if not config.ratings_path.is_file() or not config.preprocessing_report_path.is_file():
        raise FileNotFoundError("Processed ratings and preprocessing report must exist")
    fractions = (
        config.validation_user_fraction,
        config.test_user_fraction,
        config.target_fraction,
    )
    if any(not 0 < value < 1 for value in fractions):
        raise ValueError("All split fractions must be between zero and one")
    if config.validation_user_fraction + config.test_user_fraction >= 1:
        raise ValueError("Held-out user fractions must sum to less than one")
    if min(
        config.minimum_target_interactions,
        config.minimum_history_interactions,
        config.user_buckets,
        config.batch_size,
    ) <= 0:
        raise ValueError("Split minimums and batch settings must be positive")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_directory}")
    if config.working_directory.exists():
        raise FileExistsError(f"Working directory already exists: {config.working_directory}")
    report = json.loads(config.preprocessing_report_path.read_text(encoding="utf-8"))
    if _sha256(config.ratings_path) != report["output"]["ratings_sha256"]:
        raise ValueError("Processed ratings fingerprint does not match preprocessing report")
    return report


def _partition(config: UserHoldoutConfig, bucket_directory: Path) -> int:
    source = pq.ParquetFile(config.ratings_path)
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=config.batch_size):
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


def create_user_holdout_split(config: UserHoldoutConfig) -> dict[str, Any]:
    preprocessing_report = _validate(config)
    users = int(preprocessing_report["output"]["users"])
    user_groups = assign_user_groups(
        users,
        config.validation_user_fraction,
        config.test_user_fraction,
        config.seed,
    )
    config.output_directory.mkdir(parents=True, exist_ok=True)
    bucket_directory = config.working_directory / "user-buckets"
    bucket_directory.mkdir(parents=True)
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = {name: 0 for name in OUTPUT_NAMES}
    user_sets: dict[str, set[int]] = {name: set() for name in OUTPUT_NAMES}
    try:
        partitioned_rows = _partition(config, bucket_directory)
        for bucket_path in sorted(bucket_directory.glob("bucket-*.parquet")):
            frame = pq.read_table(bucket_path).to_pandas()
            labelled = label_user_rows(
                frame,
                user_groups,
                config.target_fraction,
                config.minimum_target_interactions,
                config.minimum_history_interactions,
            )
            for name in OUTPUT_NAMES:
                selected = labelled.loc[labelled["split"] == name].drop(columns="split")
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                writer = writers.get(name)
                if writer is None:
                    writer = pq.ParquetWriter(
                        config.output_directory / f"{name}.parquet",
                        table.schema,
                        compression="zstd",
                    )
                    writers[name] = writer
                writer.write_table(table)
                row_counts[name] += len(selected)
                user_sets[name].update(selected["user_index"].unique().tolist())
    finally:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(config.working_directory, ignore_errors=True)

    expected_rows = int(preprocessing_report["output"]["ratings"])
    if partitioned_rows != expected_rows or sum(row_counts.values()) != expected_rows:
        raise RuntimeError("User-holdout split row counts do not match input")
    train_users = user_sets["train"]
    validation_users = user_sets["validation_history"]
    test_users = user_sets["test_history"]
    if train_users & validation_users or train_users & test_users or validation_users & test_users:
        raise RuntimeError("User groups overlap")
    if validation_users != user_sets["validation_target"]:
        raise RuntimeError("Validation history and target users differ")
    if test_users != user_sets["test_target"]:
        raise RuntimeError("Test history and target users differ")

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
            "items": int(preprocessing_report["output"]["items"]),
        },
        "protocol": {
            "name": "strict user holdout with chronological held-out targets",
            "user_group_seed": config.seed,
            "validation_user_fraction": config.validation_user_fraction,
            "test_user_fraction": config.test_user_fraction,
            "target_fraction": config.target_fraction,
            "minimum_target_interactions": config.minimum_target_interactions,
            "minimum_history_interactions": config.minimum_history_interactions,
            "global_training_users": "train group only",
        },
        "user_groups": {
            "train": len(train_users),
            "validation": len(validation_users),
            "test": len(test_users),
            "pairwise_disjoint": True,
        },
        "output": {"files": files},
    }
    report_path = config.output_directory / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict user-holdout splits")
    parser.add_argument("--config", type=Path, default=Path("configs/user_holdout_split.toml"))
    args = parser.parse_args()
    print(
        json.dumps(
            create_user_holdout_split(UserHoldoutConfig.from_toml(args.config.resolve())),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
