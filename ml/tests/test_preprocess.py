import json
from pathlib import Path

import pandas as pd

from anime_taste_dna.data.preprocess import PreprocessConfig, preprocess_dataset


def test_preprocess_filters_and_anonymizes_interactions(tmp_path: Path) -> None:
    titles = tmp_path / "titles.csv"
    ratings = tmp_path / "ratings.csv"
    titles.write_text(
        "anime_id,title,mean,genres,studios,synopsis,media_type,num_episodes\n"
        "1,One,8.0,Action,Studio,Summary,TV,12\n"
        "2,Two,7.0,Drama,Studio,Summary,Movie,1\n"
        "3,Three,6.0,Comedy,Studio,Summary,OVA,2\n"
        "3,Three,6.0,Comedy,Studio,Summary,OVA,2\n",
        encoding="utf-8",
    )
    ratings.write_text(
        "user_id,anime_id,title,user_status,user_score,user_eps_watched,user_rewatch,updated_at\n"
        "alice,1,One,completed,9,12,False,2023-01-01T00:00:00Z\n"
        "alice,2,Two,dropped,5,1,False,2023-01-02T00:00:00Z\n"
        "bob,1,One,completed,8,12,False,2023-01-03T00:00:00Z\n"
        "bob,2,Two,completed,7,1,False,2023-01-04T00:00:00Z\n"
        "carol,3,Three,completed,6,2,False,2023-01-05T00:00:00Z\n"
        "dave,999,Missing,completed,8,1,False,2023-01-06T00:00:00Z\n"
        "eve,1,One,watching,8,3,False,2023-01-07T00:00:00Z\n"
        "frank,1,One,completed,8,12,False,1970-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    config = PreprocessConfig(
        dataset_reference="fixture",
        dataset_version=1,
        snapshot_date="2023-01-01",
        archive_sha256="fixture",
        ratings_csv=ratings,
        titles_csv=titles,
        output_directory=output,
        working_directory=tmp_path / "work",
        chunk_size=2,
        allowed_statuses=frozenset({"completed", "dropped"}),
        minimum_timestamp=pd.Timestamp("2004-01-01T00:00:00Z"),
        minimum_user_interactions=2,
        minimum_item_interactions=2,
    )

    report = preprocess_dataset(config)
    processed_ratings = pd.read_parquet(output / "ratings.parquet")
    processed_items = pd.read_parquet(output / "items.parquet")

    assert report["output"]["ratings"] == 4
    assert report["output"]["users"] == 2
    assert report["output"]["items"] == 2
    assert report["input"]["metadata"]["duplicate_mal_id_rows_removed"] == 1
    assert set(processed_ratings["user_index"]) == {0, 1}
    assert set(processed_ratings["item_index"]) == {0, 1}
    assert set(processed_ratings["mal_id"]) == {1, 2}
    assert "user_id" not in processed_ratings.columns
    assert list(processed_items["mal_id"]) == [1, 2]
    assert report["privacy"]["raw_user_ids_exported"] is False
    serialized = json.dumps(report)
    for user_name in ("alice", "bob", "carol", "dave", "eve", "frank"):
        assert user_name not in serialized
    assert not config.working_directory.exists()
