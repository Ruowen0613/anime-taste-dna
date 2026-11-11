import json
from pathlib import Path

from anime_taste_dna.data.audit import AuditConfig, audit_dataset


def test_audit_reports_quality_without_exporting_usernames(tmp_path: Path) -> None:
    titles = tmp_path / "titles.csv"
    ratings = tmp_path / "ratings.csv"
    titles.write_text(
        "anime_id,title,mean,genres,studios,synopsis,media_type,num_episodes\n"
        "1,One,8.0,Action,Studio,Summary,TV,12\n"
        "2,Two,7.0,Drama,Studio,Summary,Movie,1\n",
        encoding="utf-8",
    )
    ratings.write_text(
        "user_id,anime_id,title,user_status,user_score,user_eps_watched,user_rewatch,updated_at\n"
        "alice,1,One,completed,9,12,False,2023-01-01T00:00:00Z\n"
        "alice,1,One,completed,8,12,False,2023-01-02T00:00:00Z\n"
        "bob,2,Two,dropped,4,2,False,2023-01-03T00:00:00Z\n"
        "carol,999,Missing,completed,7,1,False,1970-01-01T00:00:00Z\n"
        "bob,999,Missing,completed,11,1,False,not-a-time\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    config = AuditConfig(
        dataset_reference="fixture",
        dataset_version=1,
        snapshot_date="2023-01-01",
        archive_sha256="fixture",
        ratings_csv=ratings,
        titles_csv=titles,
        report_json=report_path,
        working_directory=tmp_path / "work",
        chunk_size=2,
    )

    report = audit_dataset(config)

    assert report["ratings"]["raw_rows"] == 5
    assert report["ratings"]["valid_rows"] == 4
    assert report["ratings"]["unique_users"] == 3
    assert report["ratings"]["duplicate_user_item_rows"] == 1
    assert report["ratings"]["invalid_score_rows"] == 1
    assert report["ratings"]["invalid_timestamp_rows"] == 1
    assert report["ratings"]["timestamp_before_2004_rows"] == 1
    assert report["ratings"]["metadata_missing_rows"] == 1
    assert report["ratings"]["metadata_missing_unique_items"] == 1
    assert report["ratings"]["status_counts"] == {"completed": 3, "dropped": 1}
    assert report["privacy"]["raw_user_ids_exported"] is False
    serialized = json.dumps(report)
    assert "alice" not in serialized
    assert "bob" not in serialized
    assert "carol" not in serialized
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not config.working_directory.exists()
