from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RATING_COLUMNS = {
    "user_id",
    "anime_id",
    "title",
    "user_status",
    "user_score",
    "user_eps_watched",
    "user_rewatch",
    "updated_at",
}
TITLE_COLUMNS = {
    "anime_id",
    "title",
    "mean",
    "genres",
    "studios",
    "synopsis",
    "media_type",
    "num_episodes",
}


@dataclass(frozen=True)
class AuditConfig:
    dataset_reference: str
    dataset_version: int
    snapshot_date: str
    archive_sha256: str
    ratings_csv: Path
    titles_csv: Path
    report_json: Path
    working_directory: Path
    chunk_size: int

    @classmethod
    def from_toml(cls, config_path: Path) -> AuditConfig:
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
            report_json=resolve(raw["paths"]["report_json"]),
            working_directory=resolve(raw["paths"]["working_directory"]),
            chunk_size=int(raw["audit"]["chunk_size"]),
        )


def _require_columns(actual: set[str], expected: set[str], source: Path) -> None:
    missing = expected - actual
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")


def _distribution(values: dict[int, int]) -> dict[str, int | float]:
    counts = np.fromiter(values.values(), dtype=np.int64)
    if counts.size == 0:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    quantiles = np.quantile(counts, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "min": int(counts.min()),
        "p25": float(round(quantiles[0], 2)),
        "median": float(round(quantiles[1], 2)),
        "p75": float(round(quantiles[2], 2)),
        "p90": float(round(quantiles[3], 2)),
        "p99": float(round(quantiles[4], 2)),
        "max": int(counts.max()),
    }


def _count_duplicate_pairs(pair_keys_path: Path, pair_count: int) -> int:
    if pair_count < 2:
        return 0
    pair_keys = np.memmap(pair_keys_path, dtype=np.uint64, mode="r+", shape=(pair_count,))
    pair_keys.sort()
    duplicate_count = int(np.count_nonzero(pair_keys[1:] == pair_keys[:-1]))
    del pair_keys
    return duplicate_count


def audit_dataset(config: AuditConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not config.ratings_csv.is_file() or not config.titles_csv.is_file():
        raise FileNotFoundError("Ratings and titles CSV files must exist before auditing")

    title_frame = pd.read_csv(config.titles_csv, usecols=lambda column: column in TITLE_COLUMNS)
    _require_columns(set(title_frame.columns), TITLE_COLUMNS, config.titles_csv)
    numeric_title_ids = pd.to_numeric(title_frame["anime_id"], errors="coerce")
    valid_title_ids = numeric_title_ids.dropna().astype(np.int64)
    metadata_item_ids = set(valid_title_ids[valid_title_ids > 0].tolist())

    config.working_directory.mkdir(parents=True, exist_ok=True)
    pair_keys_path = config.working_directory / "pair_keys.uint64"
    if pair_keys_path.exists():
        pair_keys_path.unlink()

    raw_rows = 0
    valid_rows = 0
    invalid_score_rows = 0
    invalid_anime_id_rows = 0
    missing_user_rows = 0
    invalid_timestamp_rows = 0
    timestamp_before_2004_rows = 0
    metadata_missing_rows = 0
    metadata_missing_items: set[int] = set()
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    rating_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    user_indices: dict[str, int] = {}
    user_interactions: Counter[int] = Counter()
    item_interactions: Counter[int] = Counter()

    with pair_keys_path.open("wb") as pair_file:
        chunks = pd.read_csv(
            config.ratings_csv,
            chunksize=config.chunk_size,
            usecols=lambda column: column in RATING_COLUMNS,
            dtype={"user_id": "string", "user_status": "string"},
        )
        for chunk_number, chunk in enumerate(chunks):
            if chunk_number == 0:
                _require_columns(set(chunk.columns), RATING_COLUMNS, config.ratings_csv)
            raw_rows += len(chunk)

            anime_ids = pd.to_numeric(chunk["anime_id"], errors="coerce")
            scores = pd.to_numeric(chunk["user_score"], errors="coerce")
            users = chunk["user_id"].str.strip()
            timestamps = pd.to_datetime(chunk["updated_at"], errors="coerce", utc=True)

            valid_score = scores.between(1, 10) & (scores % 1 == 0)
            valid_anime = anime_ids.notna() & (anime_ids > 0) & (anime_ids % 1 == 0)
            valid_user = users.notna() & users.ne("")
            valid_timestamp = timestamps.notna()
            valid = valid_score & valid_anime & valid_user & valid_timestamp

            invalid_score_rows += int((~valid_score).sum())
            invalid_anime_id_rows += int((~valid_anime).sum())
            missing_user_rows += int((~valid_user).sum())
            invalid_timestamp_rows += int((~valid_timestamp).sum())
            timestamp_before_2004_rows += int(
                (valid_timestamp & (timestamps < pd.Timestamp("2004-01-01", tz="UTC"))).sum()
            )
            valid_rows += int(valid.sum())

            valid_scores = scores[valid].astype(np.uint8)
            valid_anime_ids = anime_ids[valid].astype(np.int64)
            valid_users = users[valid]
            valid_timestamps = timestamps[valid]
            valid_statuses = chunk.loc[valid, "user_status"].fillna("missing").str.lower()

            rating_counts.update(valid_scores.tolist())
            status_counts.update(valid_statuses.tolist())
            missing_metadata = ~valid_anime_ids.isin(metadata_item_ids)
            metadata_missing_rows += int(missing_metadata.sum())
            metadata_missing_items.update(valid_anime_ids[missing_metadata].tolist())
            item_interactions.update(valid_anime_ids.tolist())

            chunk_min = valid_timestamps.min()
            chunk_max = valid_timestamps.max()
            if pd.notna(chunk_min) and (timestamp_min is None or chunk_min < timestamp_min):
                timestamp_min = chunk_min
            if pd.notna(chunk_max) and (timestamp_max is None or chunk_max > timestamp_max):
                timestamp_max = chunk_max

            internal_users = np.empty(len(valid_users), dtype=np.uint32)
            for position, user_name in enumerate(valid_users):
                user_index = user_indices.get(user_name)
                if user_index is None:
                    user_index = len(user_indices)
                    user_indices[user_name] = user_index
                internal_users[position] = user_index
            user_interactions.update(internal_users.tolist())

            pair_keys = (internal_users.astype(np.uint64) << np.uint64(32)) | valid_anime_ids.to_numpy(
                dtype=np.uint64
            )
            pair_keys.tofile(pair_file)

    duplicate_pairs = _count_duplicate_pairs(pair_keys_path, valid_rows)
    shutil.rmtree(config.working_directory)

    report: dict[str, Any] = {
        "dataset": {
            "reference": config.dataset_reference,
            "version": config.dataset_version,
            "snapshot_date": config.snapshot_date,
            "archive_sha256": config.archive_sha256,
        },
        "ratings": {
            "raw_rows": raw_rows,
            "valid_rows": valid_rows,
            "invalid_score_rows": invalid_score_rows,
            "invalid_anime_id_rows": invalid_anime_id_rows,
            "missing_user_rows": missing_user_rows,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "timestamp_before_2004_rows": timestamp_before_2004_rows,
            "duplicate_user_item_rows": duplicate_pairs,
            "metadata_missing_rows": metadata_missing_rows,
            "metadata_missing_unique_items": len(metadata_missing_items),
            "unique_users": len(user_indices),
            "unique_items": len(item_interactions),
            "score_mean": round(
                sum(score * count for score, count in rating_counts.items()) / valid_rows,
                4,
            ),
            "score_counts": {str(key): rating_counts[key] for key in sorted(rating_counts)},
            "status_counts": dict(sorted(status_counts.items())),
            "timestamp_min": timestamp_min.isoformat() if timestamp_min is not None else None,
            "timestamp_max": timestamp_max.isoformat() if timestamp_max is not None else None,
            "interactions_per_user": _distribution(user_interactions),
            "interactions_per_item": _distribution(item_interactions),
        },
        "titles": {
            "raw_rows": len(title_frame),
            "valid_mal_ids": len(valid_title_ids[valid_title_ids > 0]),
            "unique_mal_ids": len(metadata_item_ids),
            "duplicate_mal_id_rows": int(valid_title_ids.duplicated().sum()),
            "invalid_mal_id_rows": int(numeric_title_ids.isna().sum() + (numeric_title_ids <= 0).sum()),
        },
        "privacy": {
            "raw_user_ids_exported": False,
            "user_index_mapping_persisted": False,
        },
    }
    config.report_json.parent.mkdir(parents=True, exist_ok=True)
    config.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw ratings snapshot in chunks")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.toml"))
    args = parser.parse_args()
    report = audit_dataset(AuditConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
