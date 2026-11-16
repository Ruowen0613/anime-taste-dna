import json
from pathlib import Path

import pytest

from app.model_artifact_verification import verify_model_artifact


def test_local_production_artifact_matches_manifest() -> None:
    backend_directory = Path(__file__).resolve().parents[1]
    repository = backend_directory.parent
    artifact = (
        repository
        / "data/artifacts/signed-rating-hybrid-v2/signed_rating_hybrid.npz"
    )
    if not artifact.is_file():
        pytest.skip("Production artifact is intentionally not stored in Git")
    manifest = backend_directory / "model/signed-rating-hybrid-v2.json"

    verify_model_artifact(artifact, manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["model_version"] == "signed-rating-hybrid-v2"
