from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr


@dataclass(frozen=True)
class MatrixFactorizationConfig:
    processed_directory: Path
    split_directory: Path
    output_directory: Path
    embedding_dimension: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    seed: int
    device: str
    fold_in_regularization_candidates: tuple[float, ...]
    fold_in_selection_ndcg_tolerance: float
    user_batch_size: int
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int

    @classmethod
    def from_toml(cls, config_path: Path) -> MatrixFactorizationConfig:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        root = config_path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            processed_directory=resolve(raw["paths"]["processed_directory"]),
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_directory=resolve(raw["paths"]["output_directory"]),
            embedding_dimension=int(raw["training"]["embedding_dimension"]),
            epochs=int(raw["training"]["epochs"]),
            batch_size=int(raw["training"]["batch_size"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            early_stopping_patience=int(raw["training"]["early_stopping_patience"]),
            seed=int(raw["training"]["seed"]),
            device=raw["training"]["device"],
            fold_in_regularization_candidates=tuple(
                float(value) for value in raw["fold_in"]["regularization_candidates"]
            ),
            fold_in_selection_ndcg_tolerance=float(
                raw["fold_in"]["selection_ndcg_tolerance"]
            ),
            user_batch_size=int(raw["fold_in"]["user_batch_size"]),
            observed_ratings=tuple(int(value) for value in raw["evaluation"]["observed_ratings"]),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
        )


class ExplicitMatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dimensions: int, global_mean: float) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(users, dimensions)
        self.item_factors = nn.Embedding(items, dimensions)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (self.user_factors(users) * self.item_factors(items)).sum(dim=1)
        return (
            self.global_mean
            + self.user_bias(users).squeeze(1)
            + self.item_bias(items).squeeze(1)
            + interaction
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def affine_permutation(length: int, seed: int) -> tuple[int, int]:
    if length <= 0:
        raise ValueError("Permutation length must be positive")
    generator = random.Random(seed)
    multiplier = generator.randrange(1, length + 1)
    while math.gcd(multiplier, length) != 1:
        multiplier = generator.randrange(1, length + 1)
    return multiplier, generator.randrange(length)


def _users_for_csr(interactions: InteractionCsr) -> np.ndarray:
    return np.repeat(
        np.arange(interactions.users, dtype=np.int32), np.diff(interactions.offsets)
    )


def _rmse(
    model: ExplicitMatrixFactorization,
    interactions: InteractionCsr,
    device: torch.device,
    batch_size: int,
) -> float:
    users = _users_for_csr(interactions)
    squared_error = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            batch_users = torch.from_numpy(users[start:end]).to(device=device, dtype=torch.long)
            batch_items = torch.from_numpy(interactions.item_indices[start:end]).to(
                device=device, dtype=torch.long
            )
            ratings = torch.from_numpy(interactions.ratings[start:end]).to(
                device=device, dtype=torch.float32
            )
            errors = model(batch_users, batch_items) - ratings
            squared_error += float(errors.square().sum().cpu())
    return math.sqrt(squared_error / len(users))


def train_model(
    train: InteractionCsr,
    validation: InteractionCsr,
    items: int,
    config: MatrixFactorizationConfig,
    device: torch.device,
) -> tuple[ExplicitMatrixFactorization, list[dict[str, float | int]], int]:
    train_users = _users_for_csr(train)
    global_mean = float(train.ratings.astype(np.float64).mean())
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = ExplicitMatrixFactorization(
        users=train.users,
        items=items,
        dimensions=config.embedding_dimension,
        global_mean=global_mean,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_rmse = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    row_positions = np.arange(config.batch_size, dtype=np.int64)

    for epoch in range(1, config.epochs + 1):
        model.train()
        multiplier, offset = affine_permutation(len(train_users), config.seed + epoch)
        squared_error = 0.0
        for start in range(0, len(train_users), config.batch_size):
            length = min(config.batch_size, len(train_users) - start)
            indices = (multiplier * (row_positions[:length] + start) + offset) % len(train_users)
            users = torch.from_numpy(train_users[indices]).to(device=device, dtype=torch.long)
            items_batch = torch.from_numpy(train.item_indices[indices]).to(
                device=device, dtype=torch.long
            )
            ratings = torch.from_numpy(train.ratings[indices]).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(users, items_batch)
            loss = torch.nn.functional.mse_loss(predictions, ratings)
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach().cpu()) * length

        train_rmse = math.sqrt(squared_error / len(train_users))
        validation_rmse = _rmse(model, validation, device, config.batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_rmse": round(train_rmse, 8),
                "validation_rmse": round(validation_rmse, 8),
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _recent_observations(
    primary: InteractionCsr,
    additional: InteractionCsr | None,
    users: np.ndarray,
    observed_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    item_rows = np.empty((len(users), observed_count), dtype=np.int32)
    rating_rows = np.empty((len(users), observed_count), dtype=np.float32)
    for row, user_index in enumerate(users):
        items = primary.user_items(int(user_index))
        ratings = primary.user_ratings(int(user_index))
        if additional is not None:
            items = np.concatenate((items, additional.user_items(int(user_index))))
            ratings = np.concatenate((ratings, additional.user_ratings(int(user_index))))
        item_rows[row] = items[-observed_count:]
        rating_rows[row] = ratings[-observed_count:]
    return item_rows, rating_rows


def fold_in_users(
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    factors = item_factors[observed_items]
    ones = np.ones((*factors.shape[:2], 1), dtype=np.float32)
    design = np.concatenate((factors, ones), axis=2)
    targets = observed_ratings - global_mean - item_bias[observed_items]
    gram = np.einsum("bni,bnj->bij", design, design)
    rhs = np.einsum("bni,bn->bi", design, targets)
    penalty = np.eye(design.shape[2], dtype=np.float32) * regularization
    parameters = np.linalg.solve(gram + penalty, rhs[..., None]).squeeze(-1)
    return parameters[:, :-1], parameters[:, -1]


def evaluate_fold_in(
    model: ExplicitMatrixFactorization,
    primary_history: InteractionCsr,
    additional_history: InteractionCsr | None,
    targets: InteractionCsr,
    supported_items: np.ndarray,
    regularization: float,
    config: MatrixFactorizationConfig,
) -> dict[str, dict[str, float | int]]:
    item_factors = model.item_factors.weight.detach().cpu().numpy()
    item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
    global_mean = float(model.global_mean.cpu())
    candidate_mask = supported_items.astype(np.bool_, copy=False)
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    results: dict[str, dict[str, float | int]] = {}

    for observed_count in config.observed_ratings:
        eligible = []
        relevant_by_user: dict[int, np.ndarray] = {}
        for user_index in range(primary_history.users):
            history_length = len(primary_history.user_items(user_index))
            if additional_history is not None:
                history_length += len(additional_history.user_items(user_index))
            target_items = targets.user_items(user_index)
            target_ratings = targets.user_ratings(user_index)
            relevant = target_items[
                (target_ratings >= config.relevance_threshold) & candidate_mask[target_items]
            ]
            if history_length >= observed_count and len(relevant) > 0:
                eligible.append(user_index)
                relevant_by_user[user_index] = relevant

        recall_sum = 0.0
        ndcg_sum = 0.0
        for start in range(0, len(eligible), config.user_batch_size):
            user_batch = np.asarray(
                eligible[start : start + config.user_batch_size], dtype=np.int32
            )
            observed_items, observed_ratings = _recent_observations(
                primary_history, additional_history, user_batch, observed_count
            )
            user_factors, user_bias = fold_in_users(
                item_factors,
                item_bias,
                global_mean,
                observed_items,
                observed_ratings,
                regularization,
            )
            scores = user_factors @ item_factors.T
            scores += user_bias[:, None] + item_bias[None, :] + global_mean
            scores[:, ~candidate_mask] = -np.inf
            for row, user_index in enumerate(user_batch):
                excluded = primary_history.user_items(int(user_index))
                if additional_history is not None:
                    excluded = np.concatenate(
                        (excluded, additional_history.user_items(int(user_index)))
                    )
                scores[row, excluded] = -np.inf
            top_indices = np.argpartition(
                scores, -config.ranking_k, axis=1
            )[:, -config.ranking_k :]
            top_scores = np.take_along_axis(scores, top_indices, axis=1)
            top_order = np.argsort(-top_scores, axis=1, kind="stable")
            recommendations = np.take_along_axis(top_indices, top_order, axis=1)

            for row, user_index in enumerate(user_batch):
                relevant = set(relevant_by_user[int(user_index)].tolist())
                hit_positions = np.asarray(
                    [
                        position
                        for position, item in enumerate(recommendations[row])
                        if item in relevant
                    ],
                    dtype=np.int64,
                )
                recall_sum += len(hit_positions) / len(relevant)
                dcg = float(discounts[hit_positions].sum()) if len(hit_positions) else 0.0
                idcg = float(discounts[: min(len(relevant), config.ranking_k)].sum())
                ndcg_sum += dcg / idcg

        results[str(observed_count)] = {
            "eligible_users": len(eligible),
            "recall_at_10": round(recall_sum / len(eligible), 8),
            "ndcg_at_10": round(ndcg_sum / len(eligible), 8),
        }
    return results


def _validate_config(config: MatrixFactorizationConfig) -> dict[str, Any]:
    positive_values = (
        config.embedding_dimension,
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.early_stopping_patience,
        *config.fold_in_regularization_candidates,
        config.user_batch_size,
        config.ranking_k,
    )
    if (
        min(positive_values) <= 0
        or config.weight_decay < 0
        or config.fold_in_selection_ndcg_tolerance < 0
    ):
        raise ValueError("Training and evaluation parameters must be positive")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Matrix-factorization output exists: {config.output_directory}")
    split_report_path = config.split_directory / "split_report.json"
    if not split_report_path.is_file():
        raise FileNotFoundError("Split report does not exist")
    report = json.loads(split_report_path.read_text(encoding="utf-8"))
    for split_name in ("train", "validation", "test"):
        split_path = config.split_directory / f"{split_name}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        expected_hash = report["output"]["files"][split_name]["sha256"]
        if _sha256(split_path) != expected_hash:
            raise ValueError(f"Fingerprint mismatch for {split_name}")
    return report


def train_and_evaluate(config: MatrixFactorizationConfig) -> dict[str, Any]:
    split_report = _validate_config(config)
    users = int(split_report["source"]["users"])
    items = int(split_report["source"]["items"])
    train_path = config.split_directory / "train.parquet"
    validation_path = config.split_directory / "validation.parquet"
    test_path = config.split_directory / "test.parquet"
    train = load_interactions_csr(train_path, users, config.batch_size)
    validation = load_interactions_csr(validation_path, users, config.batch_size)
    device = _select_device(config.device)
    model, history, best_epoch = train_model(train, validation, items, config, device)
    train_item_counts = np.bincount(train.item_indices, minlength=items)
    supported_items = train_item_counts > 0
    fold_in_trials = []
    for regularization in config.fold_in_regularization_candidates:
        metrics = evaluate_fold_in(
            model,
            primary_history=train,
            additional_history=None,
            targets=validation,
            supported_items=supported_items,
            regularization=regularization,
            config=config,
        )
        mean_ndcg = float(np.mean([entry["ndcg_at_10"] for entry in metrics.values()]))
        mean_recall = float(np.mean([entry["recall_at_10"] for entry in metrics.values()]))
        fold_in_trials.append(
            {
                "regularization": regularization,
                "mean_validation_recall_at_10": round(mean_recall, 8),
                "mean_validation_ndcg_at_10": round(mean_ndcg, 8),
                "by_observed_ratings": metrics,
            }
        )
    best_ndcg = max(trial["mean_validation_ndcg_at_10"] for trial in fold_in_trials)
    near_best_ndcg = [
        trial
        for trial in fold_in_trials
        if trial["mean_validation_ndcg_at_10"]
        >= best_ndcg - config.fold_in_selection_ndcg_tolerance
    ]
    best_fold_in = max(
        near_best_ndcg,
        key=lambda trial: (
            trial["mean_validation_recall_at_10"],
            trial["mean_validation_ndcg_at_10"],
            -trial["regularization"],
        ),
    )
    selected_regularization = float(best_fold_in["regularization"])
    test = load_interactions_csr(test_path, users, config.batch_size)
    test_metrics = evaluate_fold_in(
        model,
        primary_history=train,
        additional_history=validation,
        targets=test,
        supported_items=supported_items,
        regularization=selected_regularization,
        config=config,
    )

    config.output_directory.mkdir(parents=True)
    items_frame = pd.read_parquet(config.processed_directory / "items.parquet")
    artifact_path = config.output_directory / "matrix_factorization.npz"
    np.savez_compressed(
        artifact_path,
        item_factors=model.item_factors.weight.detach().cpu().numpy(),
        item_bias=model.item_bias.weight.detach().cpu().numpy().squeeze(1),
        global_mean=np.asarray([float(model.global_mean.cpu())], dtype=np.float32),
        item_index=items_frame["item_index"].to_numpy(dtype=np.int32),
        mal_id=items_frame["mal_id"].to_numpy(dtype=np.int64),
        supported=supported_items,
    )
    report = {
        "model": {
            "name": "explicit matrix factorization",
            "loss": "mean squared error with AdamW regularization",
            "fit_data": "train only",
            "embedding_dimension": config.embedding_dimension,
            "global_train_mean": round(float(model.global_mean.cpu()), 8),
            "supported_items": int(supported_items.sum()),
            "best_epoch": best_epoch,
            "training_history": history,
        },
        "fold_in": {
            "method": "regularized least squares with frozen item factors and bias",
            "selection_metric": (
                "highest validation Recall@10 among trials within the configured "
                "tolerance of best validation NDCG@10"
            ),
            "selection_ndcg_tolerance": config.fold_in_selection_ndcg_tolerance,
            "selected_regularization": selected_regularization,
            "validation_trials": fold_in_trials,
            "observations": "most recent interactions before the test split",
        },
        "evaluation": {
            "relevance_threshold": config.relevance_threshold,
            "ranking_k": config.ranking_k,
            "candidate_set": "all model items excluding train and validation user history",
            "test_by_observed_ratings": test_metrics,
        },
        "reproducibility": {
            "seed": config.seed,
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "source": {
            "split_report_sha256": _sha256(config.split_directory / "split_report.json"),
            "train_sha256": split_report["output"]["files"]["train"]["sha256"],
            "validation_sha256": split_report["output"]["files"]["validation"]["sha256"],
            "test_sha256": split_report["output"]["files"]["test"]["sha256"],
        },
        "artifact": {
            "path": artifact_path.name,
            "sha256": _sha256(artifact_path),
        },
    }
    report_path = config.output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate matrix factorization")
    parser.add_argument("--config", type=Path, default=Path("configs/matrix_factorization.toml"))
    args = parser.parse_args()
    report = train_and_evaluate(MatrixFactorizationConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
