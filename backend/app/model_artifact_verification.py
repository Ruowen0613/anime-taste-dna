from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_model_artifact(artifact_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")
    if artifact_path.stat().st_size != int(manifest["size_bytes"]):
        raise ValueError("Model artifact size does not match manifest")
    if sha256(artifact_path) != manifest["sha256"]:
        raise ValueError("Model artifact SHA-256 does not match manifest")

    required = {
        "item_factors",
        "item_counts",
        "mal_id",
        "title",
        "supported",
        "alpha",
        "fold_in_regularization",
        "popularity_blend",
        "positive_rating_threshold",
        "signed_fold_in_regularization",
        "signed_score_weight",
        "model_version",
    }
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"Model artifact is missing fields: {sorted(missing)}")
        model_version = str(artifact["model_version"][0])
        if model_version != manifest["model_version"]:
            raise ValueError("Model version does not match manifest")
        item_count = len(artifact["mal_id"])
        if artifact["item_factors"].shape[0] != item_count:
            raise ValueError("Item factors do not match MAL-ID count")
        if len(artifact["item_counts"]) != item_count:
            raise ValueError("Item counts do not match MAL-ID count")
