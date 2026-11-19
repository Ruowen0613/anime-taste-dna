from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RATING_COLUMNS = {"userID", "animeID", "rating"}
ANIME_COLUMNS = {
    "animeID",
    "title",
    "type",
    "year",
    "score",
    "episodes",
    "mal_url",
    "genres",
}
MAL_URL_PATTERN = re.compile(r"^https?://(?:www\.)?myanimelist\.net/anime/(\d+)(?:/|$)")


@dataclass(frozen=True)
class AuditConfig:
    dataset_reference: str
    snapshot_date: str
    license_name: str
    ratings_csv: Path
    anime_csv: Path
    report_json: Path
    working_directory: Path
    chunk_size: int
    duplicate_partitions: int

    @classmethod
    def from_toml(cls, config_path: Path) -> AuditConfig:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        root = config_path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            dataset_reference=raw["dataset"]["reference"],
            snapshot_date=raw["dataset"]["snapshot_date"],
            license_name=raw["dataset"]["license"],
            ratings_csv=resolve(raw["paths"]["ratings_csv"]),
            anime_csv=resolve(raw["paths"]["anime_csv"]),
            report_json=resolve(raw["paths"]["report_json"]),
            working_directory=resolve(raw["paths"]["working_directory"]),
            chunk_size=int(raw["audit"]["chunk_size"]),
            duplicate_partitions=int(raw["audit"]["duplicate_partitions"]),
        )


def _require_columns(actual: set[str], expected: set[str], source: Path) -> None:
    missing = expected - actual
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")


def _grow_counts(counts: np.ndarray, required_size: int) -> np.ndarray:
    if required_size <= len(counts):
        return counts
    new_size = max(required_size, max(1024, len(counts) * 2))
    grown = np.zeros(new_size, dtype=np.int64)
    grown[: len(counts)] = counts
    return grown


def _distribution(counts: np.ndarray) -> dict[str, int | float]:
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    quantiles = np.quantile(nonzero, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "min": int(nonzero.min()),
        "p25": float(round(quantiles[0], 2)),
        "median": float(round(quantiles[1], 2)),
        "p75": float(round(quantiles[2], 2)),
        "p90": float(round(quantiles[3], 2)),
        "p99": float(round(quantiles[4], 2)),
        "max": int(nonzero.max()),
    }


def _count_partition_duplicates(directory: Path, partition_count: int) -> dict[str, int]:
    duplicate_rows = 0
    duplicate_groups = 0
    conflicting_groups = 0
    for partition in range(partition_count):
        path = directory / f"pairs-{partition:03d}.uint64"
        keys = np.fromfile(path, dtype=np.uint64)
        if keys.size > 1:
            keys.sort()
            pair_keys = keys >> np.uint64(4)
            group_starts = np.r_[0, np.flatnonzero(pair_keys[1:] != pair_keys[:-1]) + 1]
            group_ends = np.r_[group_starts[1:], len(pair_keys)]
            group_sizes = group_ends - group_starts
            repeated = group_sizes > 1
            duplicate_rows += int((group_sizes[repeated] - 1).sum())
            duplicate_groups += int(repeated.sum())
            first_ratings = keys[group_starts[repeated]] & np.uint64(0xF)
            last_ratings = keys[group_ends[repeated] - 1] & np.uint64(0xF)
            conflicting_groups += int(np.count_nonzero(first_ratings != last_ratings))
    return {
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "conflicting_rating_groups": conflicting_groups,
    }


def _audit_anime(anime_csv: Path) -> tuple[dict[str, Any], set[int]]:
    frame = pd.read_csv(anime_csv, usecols=lambda column: column in ANIME_COLUMNS)
    _require_columns(set(frame.columns), ANIME_COLUMNS, anime_csv)

    internal_ids = pd.to_numeric(frame["animeID"], errors="coerce")
    parsed_mal_ids = frame["mal_url"].astype("string").str.extract(MAL_URL_PATTERN, expand=False)
    mal_ids = pd.to_numeric(parsed_mal_ids, errors="coerce")
    valid_internal = internal_ids.notna() & internal_ids.gt(0) & internal_ids.mod(1).eq(0)
    valid_mal = mal_ids.notna() & mal_ids.gt(0) & mal_ids.mod(1).eq(0)
    valid_ids = set(internal_ids[valid_internal].astype(np.int64).tolist())

    return (
        {
            "raw_rows": int(len(frame)),
            "valid_internal_ids": int(valid_internal.sum()),
            "unique_internal_ids": int(internal_ids[valid_internal].nunique()),
            "duplicate_internal_id_rows": int(
                internal_ids[valid_internal].duplicated(keep=False).sum()
            ),
            "valid_mal_ids": int(valid_mal.sum()),
            "unique_mal_ids": int(mal_ids[valid_mal].nunique()),
            "duplicate_mal_id_rows": int(mal_ids[valid_mal].duplicated(keep=False).sum()),
            "missing_or_invalid_mal_url_rows": int((~valid_mal).sum()),
            "internal_id_equals_mal_id_rows": int(
                (valid_internal & valid_mal & internal_ids.eq(mal_ids)).sum()
            ),
        },
        valid_ids,
    )


def audit_dataset(config: AuditConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.duplicate_partitions <= 0:
        raise ValueError("duplicate_partitions must be positive")
    if not config.ratings_csv.is_file() or not config.anime_csv.is_file():
        raise FileNotFoundError("Ratings and anime CSV files must exist before auditing")

    anime_report, metadata_ids = _audit_anime(config.anime_csv)
    user_counts = np.zeros(0, dtype=np.int64)
    item_counts = np.zeros(0, dtype=np.int64)
    rating_tenths_counts = np.zeros(101, dtype=np.int64)
    raw_rows = 0
    valid_rows = 0
    invalid_user_rows = 0
    invalid_anime_rows = 0
    invalid_rating_rows = 0
    fractional_rating_rows = 0
    metadata_missing_rows = 0
    metadata_missing_ids: set[int] = set()
    users_monotonic = True
    previous_user_id: int | None = None
    invalid_rating_counts: dict[str, int] = {}
    rating_sum = 0.0

    if config.working_directory.exists():
        shutil.rmtree(config.working_directory)
    config.working_directory.mkdir(parents=True)
    partition_paths = [
        config.working_directory / f"pairs-{partition:03d}.uint64"
        for partition in range(config.duplicate_partitions)
    ]
    for path in partition_paths:
        path.touch()

    chunks = pd.read_csv(
        config.ratings_csv,
        chunksize=config.chunk_size,
        usecols=lambda column: column in RATING_COLUMNS,
    )
    for chunk_number, chunk in enumerate(chunks):
        if chunk_number == 0:
            _require_columns(set(chunk.columns), RATING_COLUMNS, config.ratings_csv)
        raw_rows += len(chunk)
        users = pd.to_numeric(chunk["userID"], errors="coerce")
        anime_ids = pd.to_numeric(chunk["animeID"], errors="coerce")
        ratings = pd.to_numeric(chunk["rating"], errors="coerce")

        valid_user = users.notna() & users.gt(0) & users.mod(1).eq(0)
        valid_anime = anime_ids.notna() & anime_ids.gt(0) & anime_ids.mod(1).eq(0)
        valid_rating = ratings.notna() & ratings.between(0.1, 10.0)
        valid = valid_user & valid_anime & valid_rating
        invalid_user_rows += int((~valid_user).sum())
        invalid_anime_rows += int((~valid_anime).sum())
        invalid_rating_rows += int((~valid_rating).sum())
        for value, count in ratings[~valid_rating].value_counts(dropna=False).items():
            key = "missing" if pd.isna(value) else f"{float(value):g}"
            invalid_rating_counts[key] = invalid_rating_counts.get(key, 0) + int(count)
        valid_rows += int(valid.sum())

        valid_users = users[valid].astype(np.int64).to_numpy()
        valid_items = anime_ids[valid].astype(np.int64).to_numpy()
        valid_ratings = ratings[valid].to_numpy(dtype=np.float64)
        rating_sum += float(valid_ratings.sum())
        if valid_users.size == 0:
            continue

        users_monotonic &= bool(np.all(valid_users[1:] >= valid_users[:-1]))
        if previous_user_id is not None:
            users_monotonic &= int(valid_users[0]) >= previous_user_id
        previous_user_id = int(valid_users[-1])

        user_counts = _grow_counts(user_counts, int(valid_users.max()) + 1)
        item_counts = _grow_counts(item_counts, int(valid_items.max()) + 1)
        user_counts[: len(np.bincount(valid_users))] += np.bincount(valid_users)
        item_counts[: len(np.bincount(valid_items))] += np.bincount(valid_items)

        pair_keys = (
            ((valid_users.astype(np.uint64) << np.uint64(32)) | valid_items.astype(np.uint64))
            << np.uint64(4)
        ) | valid_ratings.astype(np.uint64)
        pair_partitions = valid_users % config.duplicate_partitions
        for partition, path in enumerate(partition_paths):
            with path.open("ab") as partition_file:
                pair_keys[pair_partitions == partition].tofile(partition_file)

        tenths = np.rint(valid_ratings * 10).astype(np.int64)
        rating_tenths_counts += np.bincount(tenths, minlength=101)[:101]
        fractional_rating_rows += int(np.count_nonzero(~np.isclose(valid_ratings % 1, 0)))

        missing_metadata = ~np.isin(valid_items, np.fromiter(metadata_ids, dtype=np.int64))
        metadata_missing_rows += int(missing_metadata.sum())
        metadata_missing_ids.update(valid_items[missing_metadata].tolist())

    score_counts = {
        f"{index / 10:g}": int(count)
        for index, count in enumerate(rating_tenths_counts)
        if count
    }
    duplicate_report = _count_partition_duplicates(
        config.working_directory, config.duplicate_partitions
    )
    shutil.rmtree(config.working_directory)
    report: dict[str, Any] = {
        "dataset": {
            "reference": config.dataset_reference,
            "snapshot_date": config.snapshot_date,
            "license": config.license_name,
            "has_rating_timestamps": False,
        },
        "ratings": {
            "raw_rows": raw_rows,
            "valid_rows": valid_rows,
            "invalid_user_id_rows": invalid_user_rows,
            "invalid_anime_id_rows": invalid_anime_rows,
            "invalid_rating_rows": invalid_rating_rows,
            "invalid_rating_counts": dict(sorted(invalid_rating_counts.items())),
            "fractional_rating_rows": fractional_rating_rows,
            "unique_users": int(np.count_nonzero(user_counts)),
            "unique_items": int(np.count_nonzero(item_counts)),
            "score_mean": round(rating_sum / valid_rows, 4),
            "score_counts": score_counts,
            "metadata_missing_rows": metadata_missing_rows,
            "metadata_missing_unique_items": len(metadata_missing_ids),
            "user_ids_monotonic": users_monotonic,
            **duplicate_report,
            "duplicate_count_is_exact": True,
            "interactions_per_user": _distribution(user_counts),
            "interactions_per_item": _distribution(item_counts),
        },
        "anime": anime_report,
        "limitations": [
            "The ratings file has no timestamps, so chronological splitting is unavailable.",
            "The source combines MyAnimeList, AniList, and Kitsu ratings on a normalized scale.",
            "animeID is a dataset-local identifier; mal_url must be parsed for canonical MAL IDs.",
        ],
    }
    config.report_json.parent.mkdir(parents=True, exist_ok=True)
    config.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the 2025 User Animelist dataset")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(AuditConfig.from_toml(args.config)), indent=2))


if __name__ == "__main__":
    main()
