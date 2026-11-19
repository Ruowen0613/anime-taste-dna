from pathlib import Path

from anime_taste_dna.data.audit_user_animelist_2025 import AuditConfig, audit_dataset


def test_audit_maps_local_anime_ids_and_counts_fractional_ratings(tmp_path: Path) -> None:
    anime_csv = tmp_path / "animes.csv"
    anime_csv.write_text(
        "animeID,title,type,year,score,episodes,mal_url,genres\n"
        "1,One,TV,2020,8.0,12,https://myanimelist.net/anime/101,[]\n"
        "2,Two,MOVIE,2021,7.0,1,https://myanimelist.net/anime/202/Two,[]\n",
        encoding="utf-8",
    )
    ratings_csv = tmp_path / "ratings.csv"
    ratings_csv.write_text(
        "userID,animeID,rating\n1,1,8\n1,2,7.5\n2,1,10\n1,1,8\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = audit_dataset(
        AuditConfig(
            dataset_reference="test",
            snapshot_date="2025-01-01",
            license_name="test",
            ratings_csv=ratings_csv,
            anime_csv=anime_csv,
            report_json=report_path,
            working_directory=tmp_path / "working",
            chunk_size=2,
            duplicate_partitions=2,
        )
    )

    assert report["ratings"]["valid_rows"] == 4
    assert report["ratings"]["fractional_rating_rows"] == 1
    assert report["ratings"]["unique_users"] == 2
    assert report["ratings"]["duplicate_count_is_exact"] is True
    assert report["ratings"]["duplicate_rows_beyond_first"] == 1
    assert report["ratings"]["duplicate_groups"] == 1
    assert report["ratings"]["conflicting_rating_groups"] == 0
    assert report["ratings"]["user_ids_monotonic"] is False
    assert report["anime"]["unique_mal_ids"] == 2
    assert report["anime"]["internal_id_equals_mal_id_rows"] == 0
    assert report_path.is_file()
