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
class FixedHoldoutConfig:
    source_directory: Path
    output_directory: Path
    history_interactions: int
    target_interactions: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> FixedHoldoutConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            source_directory=resolve(raw["paths"]["source_directory"]),
            output_directory=resolve(raw["paths"]["output_directory"]),
            history_interactions=int(raw["resample"]["history_interactions"]),
            target_interactions=int(raw["resample"]["target_interactions"]),
            seed=int(raw["resample"]["seed"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fixed_sample_positions(
    interaction_count: int, history_count: int, target_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    required = history_count + target_count
    if interaction_count < required:
        return None
    selected = np.random.default_rng(seed).permutation(interaction_count)[:required]
    return selected[:history_count], selected[history_count:]


def _resample_group(
    config: FixedHoldoutConfig,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    history = pq.read_table(config.source_directory / f"{prefix}_history.parquet").to_pandas()
    target = pq.read_table(config.source_directory / f"{prefix}_target.parquet").to_pandas()
    combined = pd.concat([history, target], ignore_index=True)
    combined = combined.sort_values(["user_index", "item_index"], kind="mergesort")
    history_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    retained_users = 0
    for user_index, positions in combined.groupby("user_index", sort=False).indices.items():
        user_rows = combined.iloc[np.asarray(positions, dtype=np.int64)].reset_index(drop=True)
        sampled = fixed_sample_positions(
            len(user_rows),
            config.history_interactions,
            config.target_interactions,
            config.seed ^ int(user_index),
        )
        if sampled is None:
            continue
        history_positions, target_positions = sampled
        history_frames.append(user_rows.iloc[history_positions])
        target_frames.append(user_rows.iloc[target_positions])
        retained_users += 1
    if not history_frames or not target_frames:
        raise RuntimeError(f"No eligible {prefix} users for fixed resampling")
    return (
        pd.concat(history_frames, ignore_index=True),
        pd.concat(target_frames, ignore_index=True),
        retained_users,
    )


def resample_fixed_user_holdout(config: FixedHoldoutConfig) -> dict[str, Any]:
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_directory}")
    if config.history_interactions <= 0 or config.target_interactions <= 0:
        raise ValueError("Fixed history and target counts must be positive")
    source_report_path = config.source_directory / "split_report.json"
    if not source_report_path.is_file():
        raise FileNotFoundError("Source split report is missing")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))

    config.output_directory.mkdir(parents=True)
    shutil.copyfile(
        config.source_directory / "train.parquet",
        config.output_directory / "train.parquet",
    )
    retained: dict[str, int] = {}
    for prefix in ("validation", "test"):
        history, target, users = _resample_group(config, prefix)
        retained[prefix] = users
        pq.write_table(
            pa.Table.from_pandas(history, preserve_index=False),
            config.output_directory / f"{prefix}_history.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pandas(target, preserve_index=False),
            config.output_directory / f"{prefix}_target.parquet",
            compression="zstd",
        )

    files: dict[str, dict[str, int | str]] = {}
    for name in OUTPUT_NAMES:
        path = config.output_directory / f"{name}.parquet"
        metadata = pq.read_metadata(path)
        files[name] = {
            "rows": metadata.num_rows,
            "sha256": _sha256(path),
        }
    files["train"]["users"] = int(source_report["user_groups"]["train"])
    for prefix in ("validation", "test"):
        files[f"{prefix}_history"]["users"] = retained[prefix]
        files[f"{prefix}_target"]["users"] = retained[prefix]

    report = {
        "source": source_report["source"],
        "protocol": {
            "name": "strict user holdout with fixed deterministic random targets",
            "source_split": source_report["protocol"]["name"],
            "resample_seed": config.seed,
            "history_interactions_per_user": config.history_interactions,
            "target_interactions_per_user": config.target_interactions,
            "global_training_users": "unchanged train group only",
        },
        "user_groups": {
            "train": int(source_report["user_groups"]["train"]),
            "validation": retained["validation"],
            "test": retained["test"],
            "pairwise_disjoint": True,
        },
        "output": {"files": files},
    }
    (config.output_directory / "split_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix held-out history and target sizes")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = FixedHoldoutConfig.from_toml(args.config)
    print(json.dumps(resample_fixed_user_holdout(config), indent=2))


if __name__ == "__main__":
    main()
