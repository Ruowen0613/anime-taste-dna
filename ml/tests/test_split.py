import hashlib
import json
from pathlib import Path

import pandas as pd

from anime_taste_dna.data.split import SplitConfig, split_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_dataset_uses_per_user_chronological_holdout(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.parquet"
    rows = []
    for user_index in range(2):
        for position in range(25):
            rows.append(
                {
                    "user_index": user_index,
                    "item_index": position,
                    "mal_id": position + 1,
                    "rating": (position % 10) + 1,
                    "status": "completed",
                    "updated_at": pd.Timestamp("2023-01-01T00:00:00Z")
                    + pd.Timedelta(days=position),
                }
            )
    frame = pd.DataFrame(reversed(rows))
    frame.to_parquet(ratings_path, index=False)
    report_path = tmp_path / "preprocessing_report.json"
    report_path.write_text(
        json.dumps(
            {
                "output": {
                    "ratings_sha256": _sha256(ratings_path),
                    "ratings": 50,
                    "users": 2,
                    "items": 25,
                }
            }
        ),
        encoding="utf-8",
    )
    config = SplitConfig(
        ratings_path=ratings_path,
        preprocessing_report_path=report_path,
        output_directory=tmp_path / "splits",
        working_directory=tmp_path / "work",
        validation_fraction=0.10,
        test_fraction=0.10,
        minimum_validation_interactions=1,
        minimum_test_interactions=1,
        minimum_train_interactions=20,
        user_buckets=2,
        batch_size=7,
    )

    report = split_dataset(config)
    train = pd.read_parquet(config.output_directory / "train.parquet")
    validation = pd.read_parquet(config.output_directory / "validation.parquet")
    test = pd.read_parquet(config.output_directory / "test.parquet")

    assert report["output"]["files"]["train"]["rows"] == 42
    assert report["output"]["files"]["validation"]["rows"] == 4
    assert report["output"]["files"]["test"]["rows"] == 4
    for user_index in range(2):
        user_train = train[train["user_index"] == user_index]
        user_validation = validation[validation["user_index"] == user_index]
        user_test = test[test["user_index"] == user_index]
        assert len(user_train) == 21
        assert list(user_validation["item_index"]) == [21, 22]
        assert list(user_test["item_index"]) == [23, 24]
        assert user_train["updated_at"].max() <= user_validation["updated_at"].min()
        assert user_validation["updated_at"].max() <= user_test["updated_at"].min()
    assert not config.working_directory.exists()
