from __future__ import annotations

import argparse
import hashlib
import json
import re
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
MAL_URL_PATTERN = re.compile(r"^https?://(?:www\.)?myanimelist\.net/anime/(\d+)(?:/|$)")


@dataclass(frozen=True)
class PilotConfig:
    ratings_csv: Path
    anime_csv: Path
    audit_report: Path
    output_directory: Path
    working_directory: Path
    report_json: Path
    chunk_size: int
    partitions: int
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    target_fraction: float
    minimum_train_interactions: int
    minimum_history_interactions: int
    minimum_target_interactions: int

    @classmethod
    def from_toml(cls, path: Path) -> PilotConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            ratings_csv=resolve(raw["paths"]["ratings_csv"]),
            anime_csv=resolve(raw["paths"]["anime_csv"]),
            audit_report=resolve(raw["paths"]["audit_report"]),
            output_directory=resolve(raw["paths"]["output_directory"]),
            working_directory=resolve(raw["paths"]["working_directory"]),
            report_json=resolve(raw["paths"]["report_json"]),
            chunk_size=int(raw["preparation"]["chunk_size"]),
            partitions=int(raw["preparation"]["partitions"]),
            seed=int(raw["split"]["seed"]),
            train_fraction=float(raw["split"]["train_fraction"]),
            validation_fraction=float(raw["split"]["validation_fraction"]),
            test_fraction=float(raw["split"]["test_fraction"]),
            target_fraction=float(raw["split"]["target_fraction"]),
            minimum_train_interactions=int(raw["split"]["minimum_train_interactions"]),
            minimum_history_interactions=int(raw["split"]["minimum_history_interactions"]),
            minimum_target_interactions=int(raw["split"]["minimum_target_interactions"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hash_users(users: np.ndarray, seed: int) -> np.ndarray:
    values = users.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def assign_group(users: np.ndarray, config: PilotConfig) -> np.ndarray:
    total_fraction = config.validation_fraction + config.test_fraction + config.train_fraction
    if not 0 < total_fraction <= 1:
        raise ValueError("Selected user fractions must sum to a value in (0, 1]")
    buckets = _hash_users(users, config.seed) % np.uint64(1_000_000)
    validation_end = int(round(config.validation_fraction * 1_000_000))
    test_end = validation_end + int(round(config.test_fraction * 1_000_000))
    train_end = test_end + int(round(config.train_fraction * 1_000_000))
    groups = np.full(len(users), -1, dtype=np.int8)
    groups[buckets < validation_end] = 1
    groups[(buckets >= validation_end) & (buckets < test_end)] = 2
    groups[(buckets >= test_end) & (buckets < train_end)] = 0
    return groups


def deduplicate_sorted_interactions(
    users: np.ndarray, items: np.ndarray, ratings: np.ndarray
) -> tuple[np.ndarray, int, int]:
    order = np.lexsort((ratings, items, users))
    users = users[order]
    items = items[order]
    ratings = ratings[order]
    changes = (users[1:] != users[:-1]) | (items[1:] != items[:-1])
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    ends = np.r_[starts[1:], len(users)]
    conflicting = ratings[starts] != ratings[ends - 1]
    keep_positions = starts[~conflicting]
    removed_duplicates = int(len(users) - len(starts))
    return keep_positions, removed_duplicates, int(conflicting.sum())


def _metadata(anime_csv: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(anime_csv)
    required = {"animeID", "title", "mal_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Anime metadata is missing columns: {sorted(missing)}")
    mal_ids = pd.to_numeric(
        frame["mal_url"].astype("string").str.extract(MAL_URL_PATTERN, expand=False),
        errors="coerce",
    )
    local_ids = pd.to_numeric(frame["animeID"], errors="coerce")
    valid = (
        local_ids.notna()
        & local_ids.gt(0)
        & local_ids.mod(1).eq(0)
        & mal_ids.notna()
        & mal_ids.gt(0)
        & mal_ids.mod(1).eq(0)
    )
    items = frame.loc[valid].copy().reset_index(drop=True)
    items["source_anime_id"] = local_ids[valid].astype(np.int32).to_numpy()
    items["mal_id"] = mal_ids[valid].astype(np.int64).to_numpy()
    items["item_index"] = np.arange(len(items), dtype=np.int32)
    lookup = np.full(int(items["source_anime_id"].max()) + 1, -1, dtype=np.int32)
    lookup[items["source_anime_id"].to_numpy()] = items["item_index"].to_numpy()
    return items, lookup


def _partition_raw(config: PilotConfig, item_lookup: np.ndarray, directory: Path) -> int:
    writers: dict[int, pq.ParquetWriter] = {}
    valid_rows = 0
    try:
        chunks = pd.read_csv(
            config.ratings_csv,
            chunksize=config.chunk_size,
            usecols=["userID", "animeID", "rating"],
        )
        for chunk_number, chunk in enumerate(chunks):
            users = pd.to_numeric(chunk["userID"], errors="coerce")
            source_items = pd.to_numeric(chunk["animeID"], errors="coerce")
            ratings = pd.to_numeric(chunk["rating"], errors="coerce")
            valid = (
                users.notna()
                & users.gt(0)
                & users.mod(1).eq(0)
                & source_items.notna()
                & source_items.gt(0)
                & source_items.mod(1).eq(0)
                & source_items.lt(len(item_lookup))
                & ratings.between(1, 10)
                & ratings.mod(1).eq(0)
            )
            selected_users = users[valid].astype(np.int32).to_numpy()
            source_item_array = source_items[valid].astype(np.int32).to_numpy()
            selected_items = item_lookup[source_item_array]
            selected_ratings = ratings[valid].astype(np.uint8).to_numpy()
            known = selected_items >= 0
            selected_users = selected_users[known]
            selected_items = selected_items[known]
            selected_ratings = selected_ratings[known]
            valid_rows += len(selected_users)

            bucket_ids = selected_users % config.partitions
            for bucket in np.unique(bucket_ids):
                mask = bucket_ids == bucket
                table = pa.table(
                    {
                        "source_user_id": selected_users[mask],
                        "item_index": selected_items[mask],
                        "rating": selected_ratings[mask],
                    }
                )
                bucket_number = int(bucket)
                writer = writers.get(bucket_number)
                if writer is None:
                    writer = pq.ParquetWriter(
                        directory / f"bucket-{bucket_number:03d}.parquet",
                        table.schema,
                        compression="zstd",
                    )
                    writers[bucket_number] = writer
                writer.write_table(table)
            if (chunk_number + 1) % 25 == 0:
                print(
                    json.dumps({"partition_chunks": chunk_number + 1, "valid_rows": valid_rows}),
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()
    return valid_rows


def _append_output(
    writers: dict[str, pq.ParquetWriter],
    output_directory: Path,
    name: str,
    user_indices: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    mal_ids: np.ndarray,
) -> int:
    if len(user_indices) == 0:
        return 0
    table = pa.table(
        {
            "user_index": user_indices.astype(np.int32),
            "item_index": items.astype(np.int32),
            "mal_id": mal_ids[items].astype(np.int64),
            "rating": ratings.astype(np.uint8),
        }
    )
    writer = writers.get(name)
    if writer is None:
        writer = pq.ParquetWriter(
            output_directory / f"{name}.parquet", table.schema, compression="zstd"
        )
        writers[name] = writer
    writer.write_table(table)
    return len(user_indices)


def _process_partitions(
    config: PilotConfig, items: pd.DataFrame, partition_directory: Path
) -> tuple[dict[str, int], dict[str, int], int, int, int]:
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = {name: 0 for name in OUTPUT_NAMES}
    user_counts = {"train": 0, "validation": 0, "test": 0}
    mal_ids = items["mal_id"].to_numpy(dtype=np.int64)
    next_user_index = 0
    duplicate_rows_removed = 0
    conflicting_groups_dropped = 0
    ignored_users = 0
    try:
        for bucket_path in sorted(partition_directory.glob("bucket-*.parquet")):
            frame = pq.read_table(bucket_path).to_pandas()
            users = frame["source_user_id"].to_numpy(dtype=np.int32)
            item_indices = frame["item_index"].to_numpy(dtype=np.int32)
            ratings = frame["rating"].to_numpy(dtype=np.uint8)
            keep, removed, conflicts = deduplicate_sorted_interactions(
                users, item_indices, ratings
            )
            order = np.lexsort((ratings, item_indices, users))
            users = users[order][keep]
            item_indices = item_indices[order][keep]
            ratings = ratings[order][keep]
            duplicate_rows_removed += removed
            conflicting_groups_dropped += conflicts

            boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
            unique_users = users[boundaries[:-1]]
            groups = assign_group(unique_users, config)
            for position, source_user in enumerate(unique_users):
                start, end = boundaries[position : position + 2]
                count = int(end - start)
                group = int(groups[position])
                if group == 0 and count < config.minimum_train_interactions:
                    ignored_users += 1
                    continue
                if group in (1, 2) and count < (
                    config.minimum_history_interactions + config.minimum_target_interactions
                ):
                    ignored_users += 1
                    continue
                if group == -1:
                    ignored_users += 1
                    continue

                user_index = next_user_index
                next_user_index += 1
                selected_items = item_indices[start:end]
                selected_ratings = ratings[start:end]
                if group == 0:
                    output_users = np.full(count, user_index, dtype=np.int32)
                    row_counts["train"] += _append_output(
                        writers,
                        config.output_directory,
                        "train",
                        output_users,
                        selected_items,
                        selected_ratings,
                        mal_ids,
                    )
                    user_counts["train"] += 1
                    continue

                target_count = max(
                    config.minimum_target_interactions,
                    int(np.floor(count * config.target_fraction)),
                )
                target_count = min(target_count, count - config.minimum_history_interactions)
                generator = np.random.default_rng(config.seed ^ int(source_user))
                shuffled = generator.permutation(count)
                target_positions = shuffled[:target_count]
                history_positions = shuffled[target_count:]
                prefix = "validation" if group == 1 else "test"
                for suffix, positions in (
                    ("history", history_positions),
                    ("target", target_positions),
                ):
                    name = f"{prefix}_{suffix}"
                    output_users = np.full(len(positions), user_index, dtype=np.int32)
                    row_counts[name] += _append_output(
                        writers,
                        config.output_directory,
                        name,
                        output_users,
                        selected_items[positions],
                        selected_ratings[positions],
                        mal_ids,
                    )
                user_counts[prefix] += 1
            print(
                json.dumps(
                    {
                        "processed_bucket": bucket_path.name,
                        "selected_users": next_user_index,
                        "rows": row_counts,
                    }
                ),
                flush=True,
            )
    finally:
        for writer in writers.values():
            writer.close()
    return (
        row_counts,
        user_counts,
        next_user_index,
        duplicate_rows_removed,
        conflicting_groups_dropped,
    )


def prepare_pilot(config: PilotConfig) -> dict[str, Any]:
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_directory}")
    if config.working_directory.exists():
        raise FileExistsError(f"Working directory already exists: {config.working_directory}")
    if not config.audit_report.is_file():
        raise FileNotFoundError("The executed 2025 dataset audit is required")
    audit = json.loads(config.audit_report.read_text(encoding="utf-8"))
    if not audit["ratings"]["duplicate_count_is_exact"]:
        raise ValueError("The source audit must include exact duplicate counts")

    config.output_directory.mkdir(parents=True)
    partition_directory = config.working_directory / "user-partitions"
    partition_directory.mkdir(parents=True)
    items, item_lookup = _metadata(config.anime_csv)
    items_output = items[
        ["item_index", "mal_id", "source_anime_id", "title", "type", "year", "genres"]
    ]
    pq.write_table(
        pa.Table.from_pandas(items_output, preserve_index=False),
        config.output_directory / "items.parquet",
        compression="zstd",
    )

    try:
        valid_source_rows = _partition_raw(config, item_lookup, partition_directory)
        (
            row_counts,
            user_counts,
            selected_users,
            duplicate_rows_removed,
            conflicting_groups_dropped,
        ) = _process_partitions(config, items, partition_directory)
    finally:
        shutil.rmtree(config.working_directory, ignore_errors=True)

    missing_outputs = [
        name for name in OUTPUT_NAMES if not (config.output_directory / f"{name}.parquet").is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Pilot preparation produced no rows for: {missing_outputs}")

    files = {
        name: {
            "rows": row_counts[name],
            "sha256": _sha256(config.output_directory / f"{name}.parquet"),
        }
        for name in OUTPUT_NAMES
    }
    split_report = {
        "source": {
            "dataset": "mramazan/User-Animelist-Dataset",
            "ratings": sum(row_counts.values()),
            "users": selected_users,
            "items": len(items),
        },
        "protocol": {
            "name": "strict user holdout with deterministic random targets; no timestamps",
            "user_group_seed": config.seed,
            "train_fraction_of_source_users": config.train_fraction,
            "validation_fraction_of_source_users": config.validation_fraction,
            "test_fraction_of_source_users": config.test_fraction,
            "target_fraction": config.target_fraction,
            "minimum_history_interactions": config.minimum_history_interactions,
            "minimum_target_interactions": config.minimum_target_interactions,
            "global_training_users": "train group only",
        },
        "user_groups": {
            **user_counts,
            "pairwise_disjoint": True,
        },
        "output": {"files": files},
    }
    (config.output_directory / "split_report.json").write_text(
        json.dumps(split_report, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "source": {
            "ratings_sha256": _sha256(config.ratings_csv),
            "anime_sha256": _sha256(config.anime_csv),
            "valid_rows_before_deduplication": valid_source_rows,
        },
        "cleaning": {
            "duplicate_rows_beyond_first_removed": duplicate_rows_removed,
            "conflicting_rating_groups_dropped": conflicting_groups_dropped,
            "conflict_policy": "drop the entire user-item group when ratings disagree",
        },
        "selection": {
            "selected_users": selected_users,
            "user_groups": user_counts,
            "rows": row_counts,
            "items": len(items),
        },
        "limitations": [
            "The source lacks timestamps, so held-out targets are deterministic random samples.",
            "This is a resource-bounded pilot using a deterministic subset of source users.",
            "Platform provenance is absent, so platform-specific calibration cannot be audited.",
        ],
    }
    config.report_json.parent.mkdir(parents=True, exist_ok=True)
    config.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 2025 dataset pilot split")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_pilot(PilotConfig.from_toml(args.config)), indent=2))


if __name__ == "__main__":
    main()
