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

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    ratings_path: Path
    preprocessing_report_path: Path
    output_directory: Path
    working_directory: Path
    validation_fraction: float
    test_fraction: float
    minimum_validation_interactions: int
    minimum_test_interactions: int
    minimum_train_interactions: int
    user_buckets: int
    batch_size: int

    @classmethod
    def from_toml(cls, config_path: Path) -> SplitConfig:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        root = config_path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        processed_directory = resolve(raw["paths"]["processed_directory"])
        return cls(
            ratings_path=processed_directory / "ratings.parquet",
            preprocessing_report_path=processed_directory / "preprocessing_report.json",
            output_directory=resolve(raw["paths"]["split_directory"]),
            working_directory=resolve(raw["paths"]["split_working_directory"]),
            validation_fraction=float(raw["split"]["validation_fraction"]),
            test_fraction=float(raw["split"]["test_fraction"]),
            minimum_validation_interactions=int(
                raw["split"]["minimum_validation_interactions"]
            ),
            minimum_test_interactions=int(raw["split"]["minimum_test_interactions"]),
            minimum_train_interactions=int(raw["split"]["minimum_train_interactions"]),
            user_buckets=int(raw["split"]["user_buckets"]),
            batch_size=int(raw["audit"]["chunk_size"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: SplitConfig) -> None:
    if not config.ratings_path.is_file() or not config.preprocessing_report_path.is_file():
        raise FileNotFoundError("Processed ratings and preprocessing report must exist")
    for name, fraction in (
        ("validation_fraction", config.validation_fraction),
        ("test_fraction", config.test_fraction),
    ):
        if not 0 < fraction < 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.validation_fraction + config.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    if min(
        config.minimum_validation_interactions,
        config.minimum_test_interactions,
        config.minimum_train_interactions,
        config.user_buckets,
        config.batch_size,
    ) <= 0:
        raise ValueError("Split minimums, bucket count, and batch size must be positive")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Split output already exists: {config.output_directory}")
    if config.working_directory.exists():
        raise FileExistsError(f"Split working directory already exists: {config.working_directory}")


def _verify_input(config: SplitConfig) -> dict[str, Any]:
    report = json.loads(config.preprocessing_report_path.read_text(encoding="utf-8"))
    expected_hash = report["output"]["ratings_sha256"]
    actual_hash = _sha256(config.ratings_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Processed ratings fingerprint differs from the preprocessing report: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return report


def _partition_by_user(config: SplitConfig, bucket_directory: Path) -> int:
    source = pq.ParquetFile(config.ratings_path)
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=config.batch_size):
            frame = batch.to_pandas()
            bucket_ids = frame["user_index"].to_numpy() % config.user_buckets
            for bucket_id in np.unique(bucket_ids):
                bucket = frame.loc[bucket_ids == bucket_id]
                table = pa.Table.from_pandas(bucket, preserve_index=False)
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
                rows += len(bucket)
    finally:
        for writer in writers.values():
            writer.close()
    return rows


def _assign_user_splits(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    frame = frame.sort_values(
        ["user_index", "updated_at", "item_index"], kind="mergesort"
    ).reset_index(drop=True)
    group = frame.groupby("user_index", sort=False)["user_index"]
    positions = group.cumcount().to_numpy()
    counts = group.transform("size").to_numpy()
    validation_counts = np.maximum(
        np.floor(counts * config.validation_fraction).astype(np.int64),
        config.minimum_validation_interactions,
    )
    test_counts = np.maximum(
        np.floor(counts * config.test_fraction).astype(np.int64),
        config.minimum_test_interactions,
    )
    train_counts = counts - validation_counts - test_counts
    if (train_counts < config.minimum_train_interactions).any():
        invalid_user = frame.loc[
            train_counts < config.minimum_train_interactions, "user_index"
        ].iloc[0]
        user_index = int(invalid_user)
        raise ValueError(
            f"User {user_index} has fewer than {config.minimum_train_interactions} "
            "training interactions under the configured split"
        )

    labels = np.full(len(frame), "train", dtype=object)
    labels[positions >= train_counts] = "validation"
    labels[positions >= counts - test_counts] = "test"
    frame["split"] = labels
    return frame


def _write_splits(
    config: SplitConfig, bucket_directory: Path
) -> tuple[dict[str, int], int, dict[str, int]]:
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = {name: 0 for name in SPLIT_NAMES}
    users_per_split: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
    all_users: set[int] = set()
    try:
        for bucket_path in sorted(bucket_directory.glob("bucket-*.parquet")):
            frame = pq.read_table(bucket_path).to_pandas()
            frame = _assign_user_splits(frame, config)
            all_users.update(frame["user_index"].unique().tolist())
            for split_name in SPLIT_NAMES:
                selected = frame.loc[frame["split"] == split_name].drop(columns="split")
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                writer = writers.get(split_name)
                if writer is None:
                    writer = pq.ParquetWriter(
                        config.output_directory / f"{split_name}.parquet",
                        table.schema,
                        compression="zstd",
                    )
                    writers[split_name] = writer
                writer.write_table(table)
                row_counts[split_name] += len(selected)
                users_per_split[split_name].update(selected["user_index"].unique().tolist())
    finally:
        for writer in writers.values():
            writer.close()
    return row_counts, len(all_users), {
        name: len(users_per_split[name]) for name in SPLIT_NAMES
    }


def split_dataset(config: SplitConfig) -> dict[str, Any]:
    _validate_config(config)
    preprocessing_report = _verify_input(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    bucket_directory = config.working_directory / "user-buckets"
    bucket_directory.mkdir(parents=True)
    try:
        partitioned_rows = _partition_by_user(config, bucket_directory)
        row_counts, users, users_per_split = _write_splits(config, bucket_directory)
    finally:
        shutil.rmtree(config.working_directory, ignore_errors=True)

    expected_rows = int(preprocessing_report["output"]["ratings"])
    if partitioned_rows != expected_rows or sum(row_counts.values()) != expected_rows:
        raise RuntimeError("Split row counts do not match the processed ratings input")

    output_files = {
        name: {
            "rows": row_counts[name],
            "users": users_per_split[name],
            "sha256": _sha256(config.output_directory / f"{name}.parquet"),
        }
        for name in SPLIT_NAMES
    }
    report = {
        "source": {
            "ratings_sha256": preprocessing_report["output"]["ratings_sha256"],
            "ratings": expected_rows,
            "users": int(preprocessing_report["output"]["users"]),
            "items": int(preprocessing_report["output"]["items"]),
        },
        "protocol": {
            "name": "per-user chronological holdout",
            "order": ["updated_at", "item_index"],
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
            "rounding": "floor with configured minimums",
            "minimum_validation_interactions": config.minimum_validation_interactions,
            "minimum_test_interactions": config.minimum_test_interactions,
            "minimum_train_interactions": config.minimum_train_interactions,
            "random_seed": None,
            "deterministic": True,
        },
        "output": {"users": users, "files": output_files},
    }
    report_path = config.output_directory / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chronological per-user dataset splits")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.toml"))
    args = parser.parse_args()
    report = split_dataset(SplitConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
