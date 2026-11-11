from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from anime_taste_dna.data.audit import RATING_COLUMNS, TITLE_COLUMNS


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_reference: str
    dataset_version: int
    snapshot_date: str
    archive_sha256: str
    ratings_csv: Path
    titles_csv: Path
    output_directory: Path
    working_directory: Path
    chunk_size: int
    allowed_statuses: frozenset[str]
    minimum_timestamp: pd.Timestamp
    minimum_user_interactions: int
    minimum_item_interactions: int

    @classmethod
    def from_toml(cls, config_path: Path) -> PreprocessConfig:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        root = config_path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            dataset_reference=raw["dataset"]["reference"],
            dataset_version=int(raw["dataset"]["version"]),
            snapshot_date=raw["dataset"]["snapshot_date"],
            archive_sha256=raw["dataset"]["archive_sha256"],
            ratings_csv=resolve(raw["paths"]["ratings_csv"]),
            titles_csv=resolve(raw["paths"]["titles_csv"]),
            output_directory=resolve(raw["paths"]["processed_directory"]),
            working_directory=resolve(raw["paths"]["preprocess_working_directory"]),
            chunk_size=int(raw["audit"]["chunk_size"]),
            allowed_statuses=frozenset(raw["preprocess"]["allowed_statuses"]),
            minimum_timestamp=pd.Timestamp(raw["preprocess"]["minimum_timestamp"]),
            minimum_user_interactions=int(raw["preprocess"]["minimum_user_interactions"]),
            minimum_item_interactions=int(raw["preprocess"]["minimum_item_interactions"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rating_chunks(config: PreprocessConfig) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        config.ratings_csv,
        chunksize=config.chunk_size,
        usecols=lambda column: column in RATING_COLUMNS,
        dtype={"user_id": "string", "user_status": "string"},
    )


def _prepare_metadata(config: PreprocessConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    metadata = pd.read_csv(config.titles_csv, usecols=lambda column: column in TITLE_COLUMNS)
    missing_columns = TITLE_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Titles CSV is missing columns: {sorted(missing_columns)}")
    metadata["anime_id"] = pd.to_numeric(metadata["anime_id"], errors="coerce")
    invalid_rows = int((metadata["anime_id"].isna() | (metadata["anime_id"] <= 0)).sum())
    metadata = metadata[metadata["anime_id"].notna() & (metadata["anime_id"] > 0)].copy()
    metadata["anime_id"] = metadata["anime_id"].astype(np.int64)
    duplicate_rows = int(metadata.duplicated("anime_id").sum())
    metadata = metadata.drop_duplicates("anime_id", keep="first")
    return metadata, {
        "raw_rows": len(metadata) + duplicate_rows + invalid_rows,
        "invalid_mal_id_rows": invalid_rows,
        "duplicate_mal_id_rows_removed": duplicate_rows,
        "unique_mal_ids": len(metadata),
    }


def _write_anonymous_base(
    config: PreprocessConfig,
    metadata_ids: set[int],
    base_directory: Path,
) -> tuple[dict[str, int], np.ndarray, Counter[int], dict[str, int]]:
    user_indices: dict[str, int] = {}
    user_counts: list[int] = []
    item_counts: Counter[int] = Counter()
    stats = Counter()

    for chunk_number, chunk in enumerate(_rating_chunks(config)):
        if chunk_number == 0:
            missing_columns = RATING_COLUMNS - set(chunk.columns)
            if missing_columns:
                raise ValueError(f"Ratings CSV is missing columns: {sorted(missing_columns)}")

        stats["raw_rows"] += len(chunk)
        users = chunk["user_id"].str.strip()
        anime_ids = pd.to_numeric(chunk["anime_id"], errors="coerce")
        ratings = pd.to_numeric(chunk["user_score"], errors="coerce")
        statuses = chunk["user_status"].str.strip().str.lower()
        timestamps = pd.to_datetime(chunk["updated_at"], errors="coerce", utc=True)

        valid_user = users.notna() & users.ne("")
        valid_item = anime_ids.notna() & (anime_ids > 0) & (anime_ids % 1 == 0)
        valid_rating = ratings.between(1, 10) & (ratings % 1 == 0)
        valid_timestamp = timestamps.notna() & (timestamps >= config.minimum_timestamp)
        allowed_status = statuses.isin(config.allowed_statuses)
        known_metadata = anime_ids.isin(metadata_ids)
        keep = (
            valid_user
            & valid_item
            & valid_rating
            & valid_timestamp
            & allowed_status
            & known_metadata
        )

        stats["excluded_missing_user"] += int((~valid_user).sum())
        stats["excluded_invalid_item"] += int((~valid_item).sum())
        stats["excluded_invalid_rating"] += int((~valid_rating).sum())
        stats["excluded_invalid_or_early_timestamp"] += int((~valid_timestamp).sum())
        stats["excluded_status"] += int((~allowed_status).sum())
        stats["excluded_missing_metadata"] += int((~known_metadata).sum())
        stats["base_rows"] += int(keep.sum())

        kept_users = users[keep]
        internal_users = np.empty(len(kept_users), dtype=np.uint32)
        for position, user_name in enumerate(kept_users):
            user_index = user_indices.get(user_name)
            if user_index is None:
                user_index = len(user_indices)
                user_indices[user_name] = user_index
                user_counts.append(0)
            internal_users[position] = user_index
            user_counts[user_index] += 1

        kept_items = anime_ids[keep].astype(np.int64).to_numpy()
        item_counts.update(kept_items.tolist())
        base = pd.DataFrame(
            {
                "source_user_index": internal_users,
                "mal_id": kept_items,
                "rating": ratings[keep].astype(np.uint8).to_numpy(),
                "status": statuses[keep].to_numpy(),
                "updated_at": timestamps[keep].to_numpy(),
            }
        )
        table = pa.Table.from_pandas(base, preserve_index=False)
        pq.write_table(
            table,
            base_directory / f"part-{chunk_number:05d}.parquet",
            compression="zstd",
        )

    return user_indices, np.asarray(user_counts, dtype=np.int64), item_counts, dict(stats)


def _refine_interaction_core(
    base_directory: Path,
    initial_user_counts: np.ndarray,
    initial_item_counts: Counter[int],
    minimum_user_interactions: int,
    minimum_item_interactions: int,
    batch_size: int,
) -> tuple[np.ndarray, set[int], int]:
    dataset = ds.dataset(base_directory, format="parquet")
    eligible_users = initial_user_counts >= minimum_user_interactions
    eligible_items = {
        mal_id
        for mal_id, count in initial_item_counts.items()
        if count >= minimum_item_interactions
    }

    for iteration in range(1, 21):
        user_counts = np.zeros(len(initial_user_counts), dtype=np.int64)
        item_counts: Counter[int] = Counter()
        item_array = np.asarray(sorted(eligible_items), dtype=np.int64)

        for batch in dataset.to_batches(
            columns=["source_user_index", "mal_id"], batch_size=batch_size
        ):
            users = batch.column("source_user_index").to_numpy()
            items = batch.column("mal_id").to_numpy()
            keep = eligible_users[users] & np.isin(items, item_array)
            if not keep.any():
                continue
            user_counts += np.bincount(users[keep], minlength=len(user_counts))
            unique_items, counts = np.unique(items[keep], return_counts=True)
            item_counts.update(dict(zip(unique_items.tolist(), counts.tolist(), strict=True)))

        next_users = user_counts >= minimum_user_interactions
        next_items = {
            mal_id for mal_id, count in item_counts.items() if count >= minimum_item_interactions
        }
        if np.array_equal(next_users, eligible_users) and next_items == eligible_items:
            return next_users, next_items, iteration
        eligible_users = next_users
        eligible_items = next_items

    raise RuntimeError("Interaction filtering did not converge after 20 iterations")


def _write_processed_outputs(
    config: PreprocessConfig,
    base_directory: Path,
    metadata: pd.DataFrame,
    eligible_users: np.ndarray,
    eligible_items: set[int],
) -> tuple[int, int, int]:
    output = config.output_directory
    ratings_path = output / "ratings.parquet"
    items_path = output / "items.parquet"
    user_remap = np.full(len(eligible_users), -1, dtype=np.int32)
    user_remap[eligible_users] = np.arange(int(eligible_users.sum()), dtype=np.int32)
    sorted_items = np.asarray(sorted(eligible_items), dtype=np.int64)
    item_lookup = {mal_id: index for index, mal_id in enumerate(sorted_items.tolist())}

    dataset = ds.dataset(base_directory, format="parquet")
    writer: pq.ParquetWriter | None = None
    output_rows = 0
    try:
        for batch in dataset.to_batches(batch_size=config.chunk_size):
            frame = batch.to_pandas()
            source_users = frame["source_user_index"].to_numpy()
            mal_ids = frame["mal_id"].to_numpy()
            keep = eligible_users[source_users] & np.isin(mal_ids, sorted_items)
            if not keep.any():
                continue
            filtered = frame.loc[keep].copy()
            filtered["user_index"] = user_remap[source_users[keep]]
            filtered["item_index"] = filtered["mal_id"].map(item_lookup).astype(np.int32)
            filtered = filtered[
                ["user_index", "item_index", "mal_id", "rating", "status", "updated_at"]
            ]
            table = pa.Table.from_pandas(filtered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(ratings_path, table.schema, compression="zstd")
            writer.write_table(table)
            output_rows += len(filtered)
    finally:
        if writer is not None:
            writer.close()

    processed_items = metadata[metadata["anime_id"].isin(eligible_items)].copy()
    processed_items = processed_items.rename(columns={"anime_id": "mal_id", "mean": "source_mean"})
    processed_items["item_index"] = processed_items["mal_id"].map(item_lookup).astype(np.int32)
    processed_items["source_mean"] = pd.to_numeric(
        processed_items["source_mean"], errors="coerce"
    ).mask(lambda values: values < 0)
    processed_items = processed_items.sort_values("item_index")
    processed_items.to_parquet(items_path, index=False, compression="zstd")
    return output_rows, int(eligible_users.sum()), len(sorted_items)


def preprocess_dataset(config: PreprocessConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.minimum_user_interactions <= 0 or config.minimum_item_interactions <= 0:
        raise ValueError("Interaction thresholds must be positive")
    if not config.ratings_csv.is_file() or not config.titles_csv.is_file():
        raise FileNotFoundError("Ratings and titles CSV files must exist before preprocessing")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Processed output already exists: {config.output_directory}")
    if config.working_directory.exists():
        raise FileExistsError(
            f"Preprocessing work directory already exists: {config.working_directory}"
        )

    config.output_directory.mkdir(parents=True, exist_ok=True)
    base_directory = config.working_directory / "anonymous-base"
    base_directory.mkdir(parents=True)
    metadata, metadata_stats = _prepare_metadata(config)
    user_indices, user_counts, item_counts, base_stats = _write_anonymous_base(
        config, set(metadata["anime_id"].tolist()), base_directory
    )
    eligible_users, eligible_items, filtering_iterations = _refine_interaction_core(
        base_directory,
        user_counts,
        item_counts,
        config.minimum_user_interactions,
        config.minimum_item_interactions,
        config.chunk_size,
    )
    output_rows, output_users, output_items = _write_processed_outputs(
        config,
        base_directory,
        metadata,
        eligible_users,
        eligible_items,
    )
    del user_indices
    shutil.rmtree(config.working_directory)

    ratings_path = config.output_directory / "ratings.parquet"
    items_path = config.output_directory / "items.parquet"
    report = {
        "dataset": {
            "reference": config.dataset_reference,
            "version": config.dataset_version,
            "snapshot_date": config.snapshot_date,
            "archive_sha256": config.archive_sha256,
        },
        "filters": {
            "allowed_statuses": sorted(config.allowed_statuses),
            "minimum_timestamp": config.minimum_timestamp.isoformat(),
            "minimum_user_interactions": config.minimum_user_interactions,
            "minimum_item_interactions": config.minimum_item_interactions,
            "interaction_core_iterations": filtering_iterations,
        },
        "input": {**base_stats, "metadata": metadata_stats},
        "output": {
            "ratings": output_rows,
            "users": output_users,
            "items": output_items,
            "ratings_sha256": _sha256(ratings_path),
            "items_sha256": _sha256(items_path),
        },
        "privacy": {
            "raw_user_ids_exported": False,
            "user_index_mapping_persisted": False,
        },
    }
    report_path = config.output_directory / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the anonymous processed ratings dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.toml"))
    args = parser.parse_args()
    report = preprocess_dataset(PreprocessConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
