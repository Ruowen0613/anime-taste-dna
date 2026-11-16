from __future__ import annotations

import numpy as np


def normalize_item_factors(item_factors: np.ndarray) -> np.ndarray:
    """Return row-normalized factors while preserving unsupported zero rows."""
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    safe_norms = np.where(norms >= 1e-8, norms, 1.0)
    return (item_factors / safe_norms).astype(np.float32, copy=False)


def signed_rating_targets(
    ratings: np.ndarray, midpoint: float = 5.5, scale: float = 4.5
) -> np.ndarray:
    """Map explicit 1-10 ratings onto a bounded negative-to-positive target."""
    if scale <= 0:
        raise ValueError("Rating target scale must be positive")
    return np.clip((ratings.astype(np.float32) - midpoint) / scale, -1.0, 1.0)


def fold_in_signed_profiles(
    normalized_item_factors: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fit cold-start user profiles from every explicit rating.

    A low rating is a signed negative target rather than an absent interaction. Item
    factors remain fixed, so this operation is bounded and suitable for API inference.
    """
    if observed_items.shape != observed_ratings.shape or observed_items.ndim != 2:
        raise ValueError("Observed items and ratings must be equally shaped matrices")
    if regularization <= 0:
        raise ValueError("Fold-in regularization must be positive")

    dimensions = normalized_item_factors.shape[1]
    penalty = np.eye(dimensions, dtype=np.float32) * regularization
    targets = signed_rating_targets(observed_ratings)
    profiles = np.empty((len(observed_items), dimensions), dtype=np.float32)
    for row in range(len(observed_items)):
        design = normalized_item_factors[observed_items[row]]
        profiles[row] = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets[row],
        )
    return profiles
