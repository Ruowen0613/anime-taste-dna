from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings


class ArtifactError(RuntimeError):
    """Raised when the configured inference artifact is missing or invalid."""


@dataclass(frozen=True)
class RecommendationArtifact:
    item_factors: np.ndarray
    item_bias: np.ndarray
    global_mean: float
    mal_ids: np.ndarray
    titles: np.ndarray
    supported: np.ndarray
    fold_in_regularization: float
    model_version: str

    @classmethod
    def from_npz(cls, path: Path) -> RecommendationArtifact:
        if not path.is_file():
            raise ArtifactError(f"Recommendation artifact not found: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "item_factors",
                    "item_bias",
                    "global_mean",
                    "mal_id",
                    "title",
                    "supported",
                    "fold_in_regularization",
                    "model_version",
                }
                missing = required - set(data.files)
                if missing:
                    raise ArtifactError(f"Artifact is missing fields: {sorted(missing)}")
                artifact = cls(
                    item_factors=data["item_factors"].astype(np.float32, copy=False),
                    item_bias=data["item_bias"].astype(np.float32, copy=False),
                    global_mean=float(data["global_mean"][0]),
                    mal_ids=data["mal_id"].astype(np.int64, copy=False),
                    titles=data["title"].astype(np.str_, copy=False),
                    supported=data["supported"].astype(np.bool_, copy=False),
                    fold_in_regularization=float(data["fold_in_regularization"][0]),
                    model_version=str(data["model_version"][0]),
                )
        except (OSError, ValueError, IndexError) as error:
            raise ArtifactError("Recommendation artifact could not be loaded") from error
        artifact.validate()
        return artifact

    def validate(self) -> None:
        items = len(self.mal_ids)
        if self.item_factors.ndim != 2 or self.item_factors.shape[0] != items:
            raise ArtifactError("Item-factor shape does not match MAL IDs")
        if any(len(values) != items for values in (self.item_bias, self.titles, self.supported)):
            raise ArtifactError("Artifact item fields have inconsistent lengths")
        if len(np.unique(self.mal_ids)) != items or (self.mal_ids <= 0).any():
            raise ArtifactError("Artifact MAL IDs must be positive and unique")
        if not np.isfinite(self.item_factors).all() or not np.isfinite(self.item_bias).all():
            raise ArtifactError("Artifact model parameters must be finite")
        if not np.isfinite(self.global_mean) or self.fold_in_regularization <= 0:
            raise ArtifactError("Artifact scalar parameters are invalid")
        if not self.model_version:
            raise ArtifactError("Artifact model version is empty")


@lru_cache
def get_recommendation_artifact() -> RecommendationArtifact:
    return RecommendationArtifact.from_npz(get_settings().model_artifact_path)


@dataclass(frozen=True)
class WeightedAlsArtifact:
    item_factors: np.ndarray
    item_counts: np.ndarray
    mal_ids: np.ndarray
    titles: np.ndarray
    supported: np.ndarray
    alpha: float
    fold_in_regularization: float
    popularity_blend: float
    positive_rating_threshold: int
    model_version: str

    @classmethod
    def from_npz(cls, path: Path) -> WeightedAlsArtifact:
        if not path.is_file():
            raise ArtifactError(f"Recommendation artifact not found: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
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
                    "model_version",
                }
                missing = required - set(data.files)
                if missing:
                    raise ArtifactError(f"Artifact is missing fields: {sorted(missing)}")
                artifact = cls(
                    item_factors=data["item_factors"].astype(np.float32, copy=False),
                    item_counts=data["item_counts"].astype(np.int64, copy=False),
                    mal_ids=data["mal_id"].astype(np.int64, copy=False),
                    titles=data["title"].astype(np.str_, copy=False),
                    supported=data["supported"].astype(np.bool_, copy=False),
                    alpha=float(data["alpha"][0]),
                    fold_in_regularization=float(data["fold_in_regularization"][0]),
                    popularity_blend=float(data["popularity_blend"][0]),
                    positive_rating_threshold=int(data["positive_rating_threshold"][0]),
                    model_version=str(data["model_version"][0]),
                )
        except (OSError, ValueError, IndexError) as error:
            raise ArtifactError("Recommendation artifact could not be loaded") from error
        artifact.validate()
        return artifact

    def validate(self) -> None:
        items = len(self.mal_ids)
        if self.item_factors.ndim != 2 or self.item_factors.shape[0] != items:
            raise ArtifactError("Item-factor shape does not match MAL IDs")
        if any(
            len(values) != items
            for values in (self.item_counts, self.titles, self.supported)
        ):
            raise ArtifactError("Artifact item fields have inconsistent lengths")
        if len(np.unique(self.mal_ids)) != items or (self.mal_ids <= 0).any():
            raise ArtifactError("Artifact MAL IDs must be positive and unique")
        if not np.isfinite(self.item_factors).all() or (self.item_counts < 0).any():
            raise ArtifactError("Artifact model parameters are invalid")
        if (
            self.alpha <= 0
            or self.fold_in_regularization <= 0
            or self.popularity_blend < 0
            or not 1 <= self.positive_rating_threshold <= 10
            or not self.model_version
        ):
            raise ArtifactError("Artifact scalar parameters are invalid")


@lru_cache
def get_weighted_als_artifact() -> WeightedAlsArtifact:
    return WeightedAlsArtifact.from_npz(get_settings().model_artifact_path)
