"""Finalist promotion and leakage-safe repeated nested Confirmation CV."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .domain import EvaluationConfig, ExperimentRecord, ProgressState
from .evaluation import (
    EvaluationResult,
    _augment_scaled_training,
    _ensure_prediction_matrix,
    _make_scaler,
    augmentation_incompatibility_reason,
    rank_within_target_tasks,
)
from .metrics import RegressionMetrics, calculate_regression_metrics
from .models import NEURAL_MODEL_IDS, create_model
from .splitting import DatasetIdentity, assert_no_group_leakage, cross_validation_splits


NEURAL_MODELS = NEURAL_MODEL_IDS


@dataclass(frozen=True)
class ConfirmationConfig:
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 3
    tuning_budget: int = 8
    finalists_per_task: int = 5
    augmentation_methods: tuple[str, ...] = ()
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Confirmation outer/inner folds must be at least 2")
        if self.outer_repeats < 1 or self.tuning_budget < 1 or self.finalists_per_task < 1:
            raise ValueError("Confirmation repeats, budget and finalist cap must be positive")


@dataclass(frozen=True)
class PromotedFinalist:
    screening_result: EvaluationResult
    screening_rank: int
    model_family: str
    candidate_key: str

    @property
    def experiment(self) -> ExperimentRecord:
        return self.screening_result.experiment

    @property
    def model_id(self) -> str:
        return self.screening_result.model_id


@dataclass(frozen=True)
class TuningTrial:
    outer_repeat: int
    outer_fold: int
    trial: int
    parameters: Mapping[str, Any]
    mean_inner_nmae: float
    effective_inner_folds: int
    failed: bool = False


@dataclass(frozen=True)
class ConfirmationOuterFold:
    outer_repeat: int
    outer_fold: int
    observation_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: RegressionMetrics
    nmae: float
    selected_parameters: Mapping[str, Any]
    selected_inner_nmae: float
    trained_epochs: int | None = None


@dataclass(frozen=True)
class ConfirmationResult:
    finalist: PromotedFinalist
    augmentation_id: str
    outer_folds: tuple[ConfirmationOuterFold, ...]
    tuning_trials: tuple[TuningTrial, ...]
    metrics: RegressionMetrics
    nmae: float
    nmae_ci_low: float
    nmae_ci_high: float
    n_observations: int
    n_groups: int | None

    @property
    def target_task_key(self) -> str:
        return self.finalist.experiment.target_task.key

    @property
    def trained_epochs(self) -> tuple[int, ...]:
        return tuple(
            int(fold.trained_epochs)
            for fold in self.outer_folds
            if fold.trained_epochs is not None
        )


def _candidate_key(result: EvaluationResult) -> str:
    payload = json.dumps(
        {
            "experiment": result.experiment.stable_key,
            "model": result.model_id,
            "loss": result.loss_id,
            "scaler": result.scaler_id,
            "feature_structure": result.experiment.feature_structure.value,
            "feature_groups": result.experiment.feature_groups,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return "CAND_" + hashlib.sha256(payload).hexdigest()[:16]


def promote_finalists(
    screening_results: Sequence[EvaluationResult], max_per_task: int = 5
) -> dict[str, tuple[PromotedFinalist, ...]]:
    if max_per_task < 1:
        raise ValueError("max_per_task must be positive")
    promoted: dict[str, tuple[PromotedFinalist, ...]] = {}
    for task_key, ranked in rank_within_target_tasks(screening_results).items():
        rank_by_key: dict[str, int] = {}
        for index, row in enumerate(ranked, start=1):
            rank_by_key.setdefault(_candidate_key(row), index)
        best_classical = next((row for row in ranked if row.model_id not in NEURAL_MODELS), None)
        best_neural = next((row for row in ranked if row.model_id in NEURAL_MODELS), None)
        best_mean = next((row for row in ranked if row.model_id == "mean"), None)
        required: list[EvaluationResult | None]
        if max_per_task == 1:
            required = [ranked[0]]
        elif max_per_task == 2:
            # With only two slots, retain the leader and the missing model family.
            leader_is_neural = ranked[0].model_id in NEURAL_MODELS
            missing_family = best_classical if leader_is_neural else best_neural
            required = [ranked[0], missing_family or best_mean]
        else:
            # Default Confirmation budgets can retain the leader, explicit mean
            # comparator, and both classical/neural families.
            required = [ranked[0], best_mean, best_classical, best_neural]
        capped = []
        seen: set[str] = set()
        for row in (*required, *ranked):
            if row is None:
                continue
            key = _candidate_key(row)
            if key in seen:
                continue
            capped.append(row)
            seen.add(key)
            if len(capped) == max_per_task:
                break
        capped.sort(key=lambda row: rank_by_key[_candidate_key(row)])
        promoted[task_key] = tuple(
            PromotedFinalist(
                row,
                rank_by_key[_candidate_key(row)],
                "neural" if row.model_id in NEURAL_MODELS else "classical",
                _candidate_key(row),
            )
            for row in capped
        )
    return promoted


def _bounded_grid(grid: Sequence[Mapping[str, Any]], budget: int, random_seed: int) -> tuple[dict[str, Any], ...]:
    rows = [dict(row) for row in grid]
    if len(rows) <= budget:
        return tuple(rows)
    rng = np.random.default_rng(random_seed)
    indexes = np.sort(rng.choice(len(rows), size=budget, replace=False))
    return tuple(rows[int(index)] for index in indexes)


def model_search_space(
    model_id: str,
    n_features: int,
    n_samples: int,
    budget: int,
    random_seed: int,
) -> tuple[dict[str, Any], ...]:
    if model_id == "mean":
        grid = ({},)
    elif model_id == "ridge":
        grid = tuple({"alpha": value} for value in (0.01, 0.1, 1.0, 10.0, 100.0))
    elif model_id == "pls":
        max_components = max(1, min(n_features, n_samples - 1, 8))
        grid = tuple({"n_components": value} for value in range(1, max_components + 1))
    elif model_id == "svr_rbf":
        grid = tuple(
            {"C": C, "epsilon": epsilon, "gamma": gamma}
            for C, epsilon, gamma in itertools.product((1.0, 10.0, 100.0), (0.03, 0.1, 0.3), ("scale", 0.03, 0.1))
        )
    elif model_id in {"random_forest", "extra_trees"}:
        grid = tuple(
            {"n_estimators": trees, "min_samples_leaf": leaf, "max_features": features}
            for trees, leaf, features in itertools.product((200, 500), (1, 2, 4), (1.0, "sqrt"))
        )
    elif model_id == "cnn1d":
        grid = tuple(
            {"channels": channels, "learning_rate": rate}
            for channels, rate in itertools.product((16, 32, 64), (0.0003, 0.001, 0.003))
        )
    elif model_id == "resnet1d":
        grid = tuple(
            {
                "channels": channels,
                "blocks": blocks,
                "learning_rate": rate,
            }
            for channels, blocks, rate in itertools.product(
                (16, 32, 64),
                (1, 2, 3),
                (0.0003, 0.001, 0.003),
            )
        )
    elif model_id == "grouped_fusion":
        grid = tuple(
            {"group_width": group_width, "fusion_width": fusion_width, "learning_rate": rate}
            for group_width, fusion_width, rate in itertools.product(
                (16, 32, 64), (32, 64, 128), (0.0003, 0.001)
            )
        )
    elif model_id == "mlp_embeddings":
        grid = tuple(
            {"embedding_dim": embedding, "hidden": hidden, "learning_rate": rate}
            for embedding, hidden, rate in itertools.product(
                (4, 8, 16), (32, 64, 128), (0.0003, 0.001)
            )
        )
    elif model_id == "ft_transformer":
        grid = tuple(
            {"token_dim": token_dim, "heads": heads, "layers": layers, "learning_rate": rate}
            for token_dim, heads, layers, rate in itertools.product(
                (16, 32), (2, 4), (1, 2, 3), (0.0003, 0.001)
            )
        )
    elif model_id == "modern_nca":
        grid = tuple(
            {
                "embedding_dim": embedding,
                "hidden_dim": hidden,
                "temperature": temperature,
                "neighbor_sample_size": neighbors,
                "learning_rate": rate,
            }
            for embedding, hidden, temperature, neighbors, rate in itertools.product(
                (8, 16, 32),
                (32, 64, 128),
                (0.1, 0.2, 0.5),
                (64, 128, 256),
                (0.0003, 0.001),
            )
        )
    elif model_id == "tabm":
        grid = tuple(
            {"k": 32, "n_blocks": blocks, "d_block": width, "learning_rate": rate}
            for blocks, width, rate in itertools.product((1, 2, 3), (64, 128, 256), (0.0005, 0.002))
        )
    elif model_id == "realmlp":
        grid = tuple(
            {"learning_rate": rate, "batch_size": batch}
            for rate, batch in itertools.product((0.01, 0.04, 0.08), (64, 128, 256))
        )
    elif model_id == "catboost":
        grid = tuple(
            {"iterations": trees, "depth": depth, "learning_rate": rate, "l2_leaf_reg": regularization}
            for trees, depth, rate, regularization in itertools.product(
                (200, 500), (4, 6, 8), (0.03, 0.1), (1.0, 3.0, 10.0)
            )
        )
    elif model_id == "xgboost":
        grid = tuple(
            {"n_estimators": trees, "max_depth": depth, "learning_rate": rate, "min_child_weight": child}
            for trees, depth, rate, child in itertools.product(
                (200, 500), (2, 4, 6), (0.03, 0.1), (1.0, 5.0)
            )
        )
    elif model_id == "lightgbm":
        grid = tuple(
            {"n_estimators": trees, "num_leaves": leaves, "learning_rate": rate, "min_child_samples": child}
            for trees, leaves, rate, child in itertools.product(
                (200, 500), (7, 15, 31), (0.03, 0.1), (5, 15, 30)
            )
        )
    elif model_id in NEURAL_MODELS:
        grid = tuple(
            {"model_multiplier": width, "learning_rate": rate}
            for width, rate in itertools.product((0.75, 1.0, 1.25, 1.5), (0.0003, 0.001, 0.003))
        )
    else:
        grid = ({},)
    return _bounded_grid(grid, budget, random_seed)


def representative_parameters(result: ConfirmationResult) -> dict[str, Any]:
    """Choose the modal outer-fold parameter set, with inner NMAE as a tie-breaker."""

    grouped: dict[str, list[ConfirmationOuterFold]] = {}
    for fold in result.outer_folds:
        key = json.dumps(dict(fold.selected_parameters), sort_keys=True, ensure_ascii=False)
        grouped.setdefault(key, []).append(fold)
    if not grouped:
        return {}
    selected_key = min(
        grouped,
        key=lambda key: (
            -len(grouped[key]),
            float(np.mean([fold.selected_inner_nmae for fold in grouped[key]])),
            key,
        ),
    )
    return dict(grouped[selected_key][0].selected_parameters)


def _effective_folds(identity: DatasetIdentity, requested: int) -> int:
    available = len(np.unique(identity.group_ids)) if identity.group_ids is not None else identity.size
    folds = min(requested, int(available))
    if folds < 2:
        raise ValueError("Confirmation requires at least two independent observations/groups")
    return folds


def _macro_nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
        prediction = prediction[:, None]
    values = []
    for index in range(truth.shape[1]):
        denominator = max(float(np.mean(np.abs(truth[:, index]))), 1e-12)
        values.append(float(np.mean(np.abs(truth[:, index] - prediction[:, index]))) / denominator)
    return float(np.mean(values))


def _training_options(config: EvaluationConfig, finalist: PromotedFinalist, parameters: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    options = {
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "validation_ratio": config.validation_ratio,
        "model_multiplier": config.model_multiplier,
        "loss_id": finalist.screening_result.loss_id if finalist.screening_result.loss_id != "native" else "mse",
        "input_columns": finalist.experiment.input_columns,
        "feature_structure": finalist.experiment.feature_structure.value,
        "structure_confirmed": finalist.experiment.structure_confirmed,
    }
    model_parameters = dict(parameters)
    if finalist.model_id in NEURAL_MODELS:
        for name in ("learning_rate", "model_multiplier"):
            if name in model_parameters:
                options[name] = model_parameters.pop(name)
    return options, model_parameters


def _fit_predict(
    finalist: PromotedFinalist,
    X: np.ndarray,
    y: np.ndarray,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    parameters: Mapping[str, Any],
    augmentation_id: str,
    training_config: EvaluationConfig,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    scaler_id = finalist.screening_result.scaler_id
    x_scaler = _make_scaler(scaler_id).fit(X[train_indexes])
    y_scaler = _make_scaler(scaler_id).fit(y[train_indexes])
    X_train = x_scaler.transform(X[train_indexes])
    y_train = y_scaler.transform(y[train_indexes])
    methods = () if augmentation_id == "original" else (augmentation_id,)
    X_train, y_train = _augment_scaled_training(
        X_train, y_train, methods, finalist.experiment, training_config, seed
    )
    options, model_parameters = _training_options(training_config, finalist, parameters)
    model = create_model(
        finalist.model_id,
        seed,
        y.shape[1],
        options,
        model_parameters,
    )
    model.fit(X_train, y_train.ravel() if y.shape[1] == 1 else y_train)
    scaled = _ensure_prediction_matrix(model.predict(x_scaler.transform(X[validation_indexes])), y.shape[1])
    return y_scaler.inverse_transform(scaled), getattr(model, "trained_epochs_", None)


def confirm_finalist(
    finalist: PromotedFinalist,
    X: np.ndarray,
    y: np.ndarray,
    identity: DatasetIdentity,
    config: ConfirmationConfig,
    training_config: EvaluationConfig,
    progress_callback: Callable[[ProgressState], None] | None = None,
) -> tuple[ConfirmationResult, ...]:
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    outer_folds = _effective_folds(identity, config.outer_folds)
    outer_plan = [
        (repeat, fold_index, split)
        for repeat in range(1, config.outer_repeats + 1)
        for fold_index, split in enumerate(
            cross_validation_splits(identity, outer_folds, config.random_seed + repeat), start=1
        )
    ]
    compatible_augmentations = tuple(
        method
        for method in config.augmentation_methods
        if augmentation_incompatibility_reason(method, finalist.experiment) is None
    )
    augmentation_variants = ("original", *compatible_augmentations)
    steps_per_variant = 0
    for repeat, outer_fold, outer_split in outer_plan:
        outer_identity = DatasetIdentity(
            identity.observation_ids[outer_split.train],
            identity.group_ids[outer_split.train] if identity.group_ids is not None else None,
        )
        inner_folds = _effective_folds(outer_identity, config.inner_folds)
        parameter_count = len(
            model_search_space(
                finalist.model_id,
                features.shape[1],
                len(outer_split.train),
                config.tuning_budget,
                config.random_seed + repeat * 100 + outer_fold,
            )
        )
        steps_per_variant += parameter_count * inner_folds + 1
    total = len(augmentation_variants) * steps_per_variant
    progress = 0
    results: list[ConfirmationResult] = []
    for augmentation_id in augmentation_variants:
        evidence: list[ConfirmationOuterFold] = []
        trials: list[TuningTrial] = []
        for repeat, outer_fold, outer_split in outer_plan:
            assert_no_group_leakage(outer_split, identity)
            outer_identity = DatasetIdentity(
                identity.observation_ids[outer_split.train],
                identity.group_ids[outer_split.train] if identity.group_ids is not None else None,
            )
            inner_folds = _effective_folds(outer_identity, config.inner_folds)
            parameter_rows = model_search_space(
                finalist.model_id,
                features.shape[1],
                len(outer_split.train),
                config.tuning_budget,
                config.random_seed + repeat * 100 + outer_fold,
            )
            best_parameters: Mapping[str, Any] | None = None
            best_score = float("inf")
            for trial_index, parameters in enumerate(parameter_rows, start=1):
                scores: list[float] = []
                failed = False
                for inner_index, inner_split in enumerate(
                    cross_validation_splits(outer_identity, inner_folds, config.random_seed + repeat * 1000 + outer_fold),
                    start=1,
                ):
                    progress += 1
                    if progress_callback:
                        progress_callback(
                            ProgressState(
                                progress,
                                total,
                                finalist.experiment.display_name,
                                f"Confirmation {finalist.model_id} | {augmentation_id} | outer {repeat}.{outer_fold} | trial {trial_index}/{len(parameter_rows)} | inner {inner_index}/{inner_folds}",
                            )
                        )
                    train_indexes = outer_split.train[inner_split.train]
                    validation_indexes = outer_split.train[inner_split.test]
                    try:
                        prediction, _trained_epochs = _fit_predict(
                            finalist,
                            features,
                            targets,
                            train_indexes,
                            validation_indexes,
                            parameters,
                            augmentation_id,
                            training_config,
                            config.random_seed + repeat * 100_000 + outer_fold * 1_000 + inner_index,
                        )
                        scores.append(_macro_nmae(targets[validation_indexes], prediction))
                    except Exception:
                        failed = True
                        scores.append(float("inf"))
                score = float(np.mean(scores))
                trials.append(TuningTrial(repeat, outer_fold, trial_index, dict(parameters), score, inner_folds, failed))
                if score < best_score:
                    best_score = score
                    best_parameters = dict(parameters)
            if best_parameters is None or not np.isfinite(best_score):
                raise RuntimeError(
                    f"All tuning trials failed for {finalist.candidate_key}, outer repeat/fold {repeat}/{outer_fold}"
                )
            progress += 1
            if progress_callback:
                progress_callback(
                    ProgressState(
                        progress,
                        total,
                        finalist.experiment.display_name,
                        f"Confirmation {finalist.model_id} | {augmentation_id} | outer evidence {repeat}.{outer_fold}",
                    )
                )
            prediction, trained_epochs = _fit_predict(
                finalist,
                features,
                targets,
                outer_split.train,
                outer_split.test,
                best_parameters,
                augmentation_id,
                training_config,
                config.random_seed + 1_000_000 + repeat * 1_000 + outer_fold,
            )
            truth = targets[outer_split.test]
            metrics = calculate_regression_metrics(truth, prediction, finalist.experiment.output_columns)
            evidence.append(
                ConfirmationOuterFold(
                    repeat,
                    outer_fold,
                    tuple(str(value) for value in identity.observation_ids[outer_split.test]),
                    truth.copy(),
                    prediction,
                    metrics,
                    _macro_nmae(truth, prediction),
                    dict(best_parameters),
                    best_score,
                    trained_epochs,
                )
            )
        all_truth = np.concatenate([fold.y_true for fold in evidence])
        all_prediction = np.concatenate([fold.y_pred for fold in evidence])
        fold_nmae = np.asarray([fold.nmae for fold in evidence], dtype=float)
        mean = float(np.mean(fold_nmae))
        half_width = (
            float(student_t.ppf(0.975, df=len(fold_nmae) - 1))
            * float(np.std(fold_nmae, ddof=1))
            / np.sqrt(len(fold_nmae))
            if len(fold_nmae) > 1
            else 0.0
        )
        results.append(
            ConfirmationResult(
                finalist,
                augmentation_id,
                tuple(evidence),
                tuple(trials),
                calculate_regression_metrics(all_truth, all_prediction, finalist.experiment.output_columns),
                mean,
                max(0.0, mean - half_width),
                mean + half_width,
                identity.size,
                int(len(np.unique(identity.group_ids))) if identity.group_ids is not None else None,
            )
        )
    return tuple(results)
