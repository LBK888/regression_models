"""Fold-local screening evaluation for the first executable V4 slice."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .augmentation import (
    calibrated_feature_noise,
    c_mixup,
    foma_augmentation,
    spectral_perturbation,
    stochastic_feature_mask,
)
from .domain import EvaluationConfig, ExperimentRecord, ProgressState
from .metrics import RegressionMetrics, calculate_regression_metrics
from .models import LOSS_CONFIGURABLE_MODEL_IDS, create_model, model_configuration, model_registry
from .splitting import DatasetIdentity, assert_no_group_leakage, cross_validation_splits, holdout_split


ProgressCallback = Callable[[ProgressState], None]


@dataclass(frozen=True)
class FoldPrediction:
    fold: int
    repetition: int
    observation_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: RegressionMetrics
    trained_epochs: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    experiment: ExperimentRecord
    model_id: str
    model_name: str
    folds: tuple[FoldPrediction, ...]
    metrics: RegressionMetrics
    evidence_label: str = "Out-of-fold"
    configuration: str = "default preset"
    loss_id: str = "native"
    scaler_id: str = "standard"
    augmentation_id: str = "original"

    @property
    def target_task_key(self) -> str:
        return self.experiment.target_task.key

    @property
    def trained_epochs(self) -> tuple[int, ...]:
        return tuple(
            int(fold.trained_epochs)
            for fold in self.folds
            if fold.trained_epochs is not None
        )


def format_trained_epochs(values: Sequence[int | None], maximum: int) -> str:
    """Format actual neural epochs without implying that classical models use epochs."""
    completed = tuple(int(value) for value in values if value is not None and int(value) > 0)
    if not completed:
        return ""
    last = completed[-1]
    low, high = min(completed), max(completed)
    if low == high:
        return f"epochs={last}/{maximum}"
    return f"epochs=last {last}/{maximum}, range {low}–{high}"


def _make_scaler(scaler_id: str):
    return {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}[scaler_id]()


def augmentation_incompatibility_reason(
    method: str,
    experiment: ExperimentRecord,
) -> str | None:
    """Return an Experiment-specific reason for skipping an augmentation."""

    if method != "spectral_perturbation":
        return None
    try:
        _infer_spectral_groups(experiment.input_columns)
    except ValueError as exc:
        return str(exc)
    return None


def _augmentation_variants(
    config: EvaluationConfig,
    experiment: ExperimentRecord | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    variants: list[tuple[str, tuple[str, ...]]] = [("original", ())]
    methods = tuple(
        method
        for method in config.selected_augmentations
        if experiment is None or augmentation_incompatibility_reason(method, experiment) is None
    )
    if config.augmentation_policy == "compare_separately":
        variants.extend((method, (method,)) for method in methods)
    elif config.augmentation_policy == "combined" and methods:
        variants.append(("combined:" + "+".join(methods), methods))
    return tuple(variants)


def _configuration_specs(
    model_ids: Sequence[str],
    config: EvaluationConfig,
    experiment: ExperimentRecord | None = None,
):
    for model_id in model_ids:
        losses = config.selected_losses if model_id in LOSS_CONFIGURABLE_MODEL_IDS else ("native",)
        for scaler_id in config.selected_scalers:
            for loss_id in losses:
                for augmentation_id, methods in _augmentation_variants(config, experiment):
                    yield model_id, scaler_id, loss_id, augmentation_id, methods


def evaluation_run_count(
    model_ids: Sequence[str],
    config: EvaluationConfig,
    evidence_splits: int,
    experiment: ExperimentRecord | None = None,
) -> int:
    return (
        sum(1 for _ in _configuration_specs(model_ids, config, experiment))
        * config.repetitions
        * evidence_splits
    )


def _infer_spectral_groups(columns: Sequence[str]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Find and order every independent wavelength group; leave tabular columns untouched."""

    grouped: dict[str, list[tuple[int, float]]] = {}
    for index, column in enumerate(columns):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm$", column, re.IGNORECASE)
        if not match:
            continue
        group_name = column.rsplit("::", 1)[0] if "::" in column else "__default__"
        grouped.setdefault(group_name, []).append((index, float(match.group(1))))
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for group_name, entries in grouped.items():
        ordered = sorted(entries, key=lambda item: item[1])
        axis = np.asarray([coordinate for _index, coordinate in ordered], dtype=float)
        if len(np.unique(axis)) != len(axis):
            raise ValueError(f"Spectral group {group_name!r} contains duplicate wavelength coordinates")
        output.append((np.asarray([index for index, _coordinate in ordered], dtype=int), axis))
    if not output:
        raise ValueError("Spectral perturbation requires at least one wavelength feature such as 410nm")
    return tuple(output)


def _augment_scaled_training(
    X: np.ndarray,
    y: np.ndarray,
    methods: Sequence[str],
    experiment: ExperimentRecord,
    config: EvaluationConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not methods:
        return X, y
    count = max(1, round(len(X) * config.augmentation_ratio)) * config.augmentation_repeats
    feature_parts = [X] if config.augmentation_include_original else []
    target_parts = [y] if config.augmentation_include_original else []
    rng = np.random.default_rng(seed)
    for method_index, method in enumerate(methods):
        method_seed = seed + 1009 * (method_index + 1)
        if method == "c_mixup":
            new_X, new_y = c_mixup(X, y, count, random_seed=method_seed, alpha=config.c_mixup_alpha)
        elif method == "calibrated_noise":
            new_X, new_y = calibrated_feature_noise(
                X, y, count, random_seed=method_seed, noise_fraction=config.noise_fraction
            )
        elif method == "spectral_perturbation":
            indexes = rng.integers(0, len(X), size=count)
            new_X, new_y = X[indexes].copy(), y[indexes].copy()
            for group_index, (feature_indexes, axis) in enumerate(
                _infer_spectral_groups(experiment.input_columns)
            ):
                perturbed, _targets = spectral_perturbation(
                    X[:, feature_indexes],
                    y,
                    count,
                    random_seed=method_seed + group_index,
                    axis=axis,
                    source_indexes=indexes,
                )
                new_X[:, feature_indexes] = perturbed
        elif method == "feature_masking":
            indexes = rng.integers(0, len(X), size=count)
            masked = stochastic_feature_mask(
                X[indexes],
                random_seed=method_seed,
                probability=config.feature_mask_probability,
                add_indicators=False,
            )
            new_X, new_y = masked.features, y[indexes].copy()
        elif method == "foma":
            new_X, new_y = foma_augmentation(
                X,
                y,
                count,
                random_seed=method_seed,
                alpha=config.foma_alpha,
                k=config.foma_k,
            )
        else:
            raise ValueError(method)
        feature_parts.append(new_X)
        target_parts.append(new_y)
    return np.concatenate(feature_parts), np.concatenate(target_parts)

def _ensure_prediction_matrix(prediction: np.ndarray, n_outputs: int) -> np.ndarray:
    output = np.asarray(prediction, dtype=float)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    if output.ndim != 2 or output.shape[1] != n_outputs:
        raise ValueError(f"Model returned invalid prediction shape {output.shape}")
    return output


def evaluate_screening(
    experiment: ExperimentRecord,
    X: np.ndarray,
    y: np.ndarray,
    identity: DatasetIdentity,
    model_ids: Sequence[str],
    *,
    folds: int = 3,
    random_seed: int = 42,
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    config: EvaluationConfig | None = None,
) -> tuple[EvaluationResult, ...]:
    """Evaluate core models with fold-local X/y scaling and OOF predictions."""

    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError(f"Expected aligned 2D X/y, got {features.shape} and {targets.shape}")
    if features.shape[0] != identity.size:
        raise ValueError("DatasetIdentity length must match X/y rows")
    effective = config or EvaluationConfig(selected_losses=("mse",), augmentation_policy="original_only")
    target_names = experiment.output_columns
    capability_names = {item.model_id: item.display_name for item in model_registry()}
    split_plan = tuple(cross_validation_splits(identity, folds, random_seed))
    total = progress_total or evaluation_run_count(model_ids, effective, len(split_plan), experiment)
    results: list[EvaluationResult] = []

    current = progress_offset
    for model_id, scaler_id, loss_id, augmentation_id, augmentation_methods in _configuration_specs(
        model_ids, effective, experiment
    ):
        predictions: list[FoldPrediction] = []
        for repetition in range(1, effective.repetitions + 1):
            for fold_index, split in enumerate(split_plan, start=1):
                assert_no_group_leakage(split, identity)
                current += 1
                if progress_callback:
                    progress_callback(
                        ProgressState(
                            current,
                            total,
                            experiment.display_name,
                            f"{capability_names.get(model_id, model_id)} | {loss_id} | {scaler_id} | {augmentation_id} | rep {repetition}/{effective.repetitions} | fold {fold_index}/{len(split_plan)}",
                        )
                    )
                x_scaler = _make_scaler(scaler_id).fit(features[split.train])
                y_scaler = _make_scaler(scaler_id).fit(targets[split.train])
                X_train = x_scaler.transform(features[split.train])
                X_test = x_scaler.transform(features[split.test])
                y_train = y_scaler.transform(targets[split.train])
                seed = random_seed + 100_000 * repetition + fold_index
                X_train, y_train = _augment_scaled_training(
                    X_train, y_train, augmentation_methods, experiment, effective, seed
                )
                model = create_model(
                    model_id,
                    seed,
                    targets.shape[1],
                    {
                        "epochs": effective.epochs,
                        "patience": effective.patience,
                        "batch_size": effective.batch_size,
                        "learning_rate": effective.learning_rate,
                        "validation_ratio": effective.validation_ratio,
                        "model_multiplier": effective.model_multiplier,
                        "loss_id": loss_id if loss_id != "native" else "mse",
                        "input_columns": experiment.input_columns,
                        "feature_structure": experiment.feature_structure.value,
                        "structure_confirmed": experiment.structure_confirmed,
                    },
                )
                fit_y = y_train.ravel() if targets.shape[1] == 1 else y_train
                model.fit(X_train, fit_y)
                trained_epochs = getattr(model, "trained_epochs_", None)
                scaled_prediction = _ensure_prediction_matrix(model.predict(X_test), targets.shape[1])
                prediction = y_scaler.inverse_transform(scaled_prediction)
                truth = targets[split.test]
                metrics = calculate_regression_metrics(truth, prediction, target_names)
                predictions.append(
                    FoldPrediction(
                        fold=fold_index,
                        repetition=repetition,
                        observation_ids=tuple(str(value) for value in identity.observation_ids[split.test]),
                        y_true=truth.copy(),
                        y_pred=prediction,
                        metrics=metrics,
                        trained_epochs=trained_epochs,
                    )
                )
        all_truth = np.concatenate([item.y_true for item in predictions], axis=0)
        all_prediction = np.concatenate([item.y_pred for item in predictions], axis=0)
        results.append(
            EvaluationResult(
                experiment=experiment,
                model_id=model_id,
                model_name=capability_names.get(model_id, model_id),
                folds=tuple(predictions),
                metrics=calculate_regression_metrics(all_truth, all_prediction, target_names),
                evidence_label="Out-of-fold",
                configuration=f"{model_configuration(model_id)}; loss={loss_id}; scaler={scaler_id}; augmentation={augmentation_id}; repetitions={effective.repetitions}",
                loss_id=loss_id,
                scaler_id=scaler_id,
                augmentation_id=augmentation_id,
            )
        )
    return tuple(results)


def evaluate_holdout(
    experiment: ExperimentRecord,
    X: np.ndarray,
    y: np.ndarray,
    identity: DatasetIdentity,
    model_ids: Sequence[str],
    *,
    test_size: float,
    random_seed: int = 42,
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    config: EvaluationConfig | None = None,
) -> tuple[EvaluationResult, ...]:
    """Evaluate once on a leakage-aware holdout; test_size=0 has no metrics."""

    if test_size == 0:
        raise ValueError("A 0% holdout has no evaluation rows; use fit_final_models instead")
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError(f"Expected aligned 2D X/y, got {features.shape} and {targets.shape}")
    if len(features) != identity.size:
        raise ValueError("DatasetIdentity length must match X/y rows")
    effective = config or EvaluationConfig(selected_losses=("mse",))
    split = holdout_split(identity, test_size, random_seed)
    assert_no_group_leakage(split, identity)
    names = {item.model_id: item.display_name for item in model_registry()}
    total = progress_total or evaluation_run_count(model_ids, effective, 1, experiment)
    results: list[EvaluationResult] = []
    current = progress_offset
    for model_id, scaler_id, loss_id, augmentation_id, augmentation_methods in _configuration_specs(
        model_ids, effective, experiment
    ):
        predictions: list[FoldPrediction] = []
        for repetition in range(1, effective.repetitions + 1):
            current += 1
            if progress_callback:
                progress_callback(
                    ProgressState(
                        current,
                        total,
                        experiment.display_name,
                        f"{names.get(model_id, model_id)} | {loss_id} | {scaler_id} | {augmentation_id} | rep {repetition}/{effective.repetitions} | holdout test",
                    )
                )
            x_scaler = _make_scaler(scaler_id).fit(features[split.train])
            y_scaler = _make_scaler(scaler_id).fit(targets[split.train])
            X_train = x_scaler.transform(features[split.train])
            y_train = y_scaler.transform(targets[split.train])
            seed = random_seed + 100_000 * repetition
            X_train, y_train = _augment_scaled_training(
                X_train, y_train, augmentation_methods, experiment, effective, seed
            )
            model = create_model(
                model_id,
                seed,
                targets.shape[1],
                {
                    "epochs": effective.epochs,
                    "patience": effective.patience,
                    "batch_size": effective.batch_size,
                    "learning_rate": effective.learning_rate,
                    "validation_ratio": effective.validation_ratio,
                    "model_multiplier": effective.model_multiplier,
                    "loss_id": loss_id if loss_id != "native" else "mse",
                    "input_columns": experiment.input_columns,
                    "feature_structure": experiment.feature_structure.value,
                    "structure_confirmed": experiment.structure_confirmed,
                },
            )
            model.fit(X_train, y_train.ravel() if targets.shape[1] == 1 else y_train)
            trained_epochs = getattr(model, "trained_epochs_", None)
            scaled_prediction = _ensure_prediction_matrix(
                model.predict(x_scaler.transform(features[split.test])), targets.shape[1]
            )
            prediction = y_scaler.inverse_transform(scaled_prediction)
            truth = targets[split.test]
            metrics = calculate_regression_metrics(truth, prediction, experiment.output_columns)
            predictions.append(
                FoldPrediction(
                    fold=1,
                    repetition=repetition,
                    observation_ids=tuple(str(value) for value in identity.observation_ids[split.test]),
                    y_true=truth.copy(),
                    y_pred=prediction,
                    metrics=metrics,
                    trained_epochs=trained_epochs,
                )
            )
        all_truth = np.concatenate([item.y_true for item in predictions])
        all_prediction = np.concatenate([item.y_pred for item in predictions])
        metrics = calculate_regression_metrics(all_truth, all_prediction, experiment.output_columns)
        results.append(
            EvaluationResult(
                experiment,
                model_id,
                names.get(model_id, model_id),
                tuple(predictions),
                metrics,
                "Holdout test",
                f"{model_configuration(model_id)}; loss={loss_id}; scaler={scaler_id}; augmentation={augmentation_id}; repetitions={effective.repetitions}",
                loss_id,
                scaler_id,
                augmentation_id,
            )
        )
    return tuple(results)


def evaluate_external_model(
    selected_result: EvaluationResult,
    development_X: np.ndarray,
    development_y: np.ndarray,
    external_X: np.ndarray,
    external_y: np.ndarray,
    external_identity: DatasetIdentity,
    *,
    random_seed: int = 42,
    config: EvaluationConfig | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    selected_augmentation_id: str | None = None,
    augmentation_methods: Sequence[str] = (),
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[EvaluationResult, "FittedModelBundle"]:
    """Fit a pre-selected configuration on development data and score external rows once.

    The external targets never participate in scaling, augmentation, model fitting, or
    configuration selection.  Keeping this operation separate from screening makes the
    independent-evidence boundary explicit and testable.
    """

    external_features = np.asarray(external_X, dtype=float)
    external_targets = np.asarray(external_y, dtype=float)
    if external_targets.ndim == 1:
        external_targets = external_targets.reshape(-1, 1)
    if external_features.ndim != 2 or external_targets.ndim != 2:
        raise ValueError("External X/y must both be two-dimensional")
    if len(external_features) != len(external_targets) or len(external_features) != external_identity.size:
        raise ValueError("External X/y/identity lengths must match")
    if external_targets.shape[1] != len(selected_result.experiment.output_columns):
        raise ValueError("External target count does not match the selected Target Task")

    effective = config or EvaluationConfig(selected_losses=("mse",))
    final_parameters = dict(model_parameters or {})
    external_config = EvaluationConfig(
        **{
            **effective.__dict__,
            "repetitions": 1,
            "selected_scalers": (selected_result.scaler_id,),
            "selected_losses": (
                (selected_result.loss_id,)
                if selected_result.loss_id != "native"
                else ("mse",)
            ),
            "augmentation_policy": "original_only",
            "selected_augmentations": (),
            "learning_rate": float(
                final_parameters.pop("learning_rate", effective.learning_rate)
            ),
            "model_multiplier": float(
                final_parameters.pop("model_multiplier", effective.model_multiplier)
            ),
        }
    )
    augmentation_id = selected_augmentation_id or selected_result.augmentation_id
    bundles = fit_final_models(
        selected_result.experiment,
        development_X,
        development_y,
        (selected_result.model_id,),
        random_seed=random_seed,
        evidence_note="Selected without external targets, fit on all development rows, then evaluated once externally.",
        progress_callback=progress_callback,
        progress_offset=progress_offset,
        progress_total=progress_total,
        config=external_config,
        model_parameters=final_parameters,
        augmentation_override=(augmentation_id, tuple(augmentation_methods)),
    )
    if len(bundles) != 1:
        raise RuntimeError("External evaluation expected exactly one fitted model bundle")
    bundle = bundles[0]
    prediction = bundle.predict(external_features)
    metrics = calculate_regression_metrics(
        external_targets,
        prediction,
        selected_result.experiment.output_columns,
    )
    fold = FoldPrediction(
        fold=1,
        repetition=1,
        observation_ids=tuple(str(value) for value in external_identity.observation_ids),
        y_true=external_targets.copy(),
        y_pred=prediction,
        metrics=metrics,
        trained_epochs=bundle.trained_epochs,
    )
    result = EvaluationResult(
        experiment=selected_result.experiment,
        model_id=selected_result.model_id,
        model_name=selected_result.model_name,
        folds=(fold,),
        metrics=metrics,
        evidence_label="External validation",
        configuration=(
            f"pre-selected from development evidence; {model_configuration(selected_result.model_id)}; "
            f"loss={selected_result.loss_id}; scaler={selected_result.scaler_id}; "
            f"augmentation={augmentation_id}"
        ),
        loss_id=selected_result.loss_id,
        scaler_id=selected_result.scaler_id,
        augmentation_id=augmentation_id,
    )
    return result, bundle


@dataclass
class FittedModelBundle:
    experiment: ExperimentRecord
    model_id: str
    model_name: str
    model: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    training_rows: int
    evidence_note: str
    loss_id: str = "native"
    scaler_id: str = "standard"
    augmentation_id: str = "original"
    repetition: int = 1
    model_parameters: Mapping[str, Any] | None = None

    @property
    def trained_epochs(self) -> int | None:
        value = getattr(self.model, "trained_epochs_", None)
        return int(value) if value is not None else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        scaled = _ensure_prediction_matrix(
            self.model.predict(self.x_scaler.transform(features)), len(self.experiment.output_columns)
        )
        return self.y_scaler.inverse_transform(scaled)


def fit_final_models(
    experiment: ExperimentRecord,
    X: np.ndarray,
    y: np.ndarray,
    model_ids: Sequence[str],
    *,
    random_seed: int = 42,
    evidence_note: str = "Final fit is not generalization evidence.",
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    config: EvaluationConfig | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    augmentation_override: tuple[str, tuple[str, ...]] | None = None,
) -> tuple[FittedModelBundle, ...]:
    """Fit deployable bundles on every row without reporting training scores as evidence."""

    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    effective = config or EvaluationConfig(selected_losses=("mse",))
    names = {item.model_id: item.display_name for item in model_registry()}
    total = progress_total or evaluation_run_count(model_ids, effective, 1, experiment)
    bundles: list[FittedModelBundle] = []
    current = progress_offset
    configuration_rows = tuple(_configuration_specs(model_ids, effective, experiment))
    if augmentation_override is not None:
        selected_id, selected_methods = augmentation_override
        configuration_rows = tuple(
            (model_id, scaler_id, loss_id, selected_id, selected_methods)
            for model_id, scaler_id, loss_id, _augmentation_id, _methods in configuration_rows
        )
    for model_id, scaler_id, loss_id, augmentation_id, augmentation_methods in configuration_rows:
        for repetition in range(1, effective.repetitions + 1):
            current += 1
            if progress_callback:
                progress_callback(
                    ProgressState(
                        current,
                        total,
                        experiment.display_name,
                        f"{names.get(model_id, model_id)} | {loss_id} | {scaler_id} | {augmentation_id} | rep {repetition} | final fit",
                    )
                )
            x_scaler = _make_scaler(scaler_id).fit(features)
            y_scaler = _make_scaler(scaler_id).fit(targets)
            scaled_X = x_scaler.transform(features)
            scaled_y = y_scaler.transform(targets)
            seed = random_seed + 100_000 * repetition
            scaled_X, scaled_y = _augment_scaled_training(
                scaled_X, scaled_y, augmentation_methods, experiment, effective, seed
            )
            model = create_model(
                model_id,
                seed,
                targets.shape[1],
                {
                    "epochs": effective.epochs,
                    "patience": effective.patience,
                    "batch_size": effective.batch_size,
                    "learning_rate": effective.learning_rate,
                    "validation_ratio": effective.validation_ratio,
                    "model_multiplier": effective.model_multiplier,
                    "loss_id": loss_id if loss_id != "native" else "mse",
                    "input_columns": experiment.input_columns,
                    "feature_structure": experiment.feature_structure.value,
                    "structure_confirmed": experiment.structure_confirmed,
                },
                dict(model_parameters or {}),
            )
            model.fit(scaled_X, scaled_y.ravel() if targets.shape[1] == 1 else scaled_y)
            bundles.append(
                FittedModelBundle(
                    experiment,
                    model_id,
                    names.get(model_id, model_id),
                    model,
                    x_scaler,
                    y_scaler,
                    len(features),
                    evidence_note,
                    loss_id,
                    scaler_id,
                    augmentation_id,
                    repetition,
                    dict(model_parameters or {}),
                )
            )
    return tuple(bundles)


MODEL_KEEP_LIMIT = 3
"""How many ranked models per Target Task are persisted as deployable bundles."""

MODEL_KEEP_MIN_REPORTED_R2 = 0.0
"""A bundle is only persisted when its Reported R2 is strictly above this value."""


def select_retained_results(
    results: Sequence[EvaluationResult],
    *,
    limit: int = MODEL_KEEP_LIMIT,
    minimum_reported_r2: float = MODEL_KEEP_MIN_REPORTED_R2,
) -> tuple[dict[str, tuple[tuple[int, EvaluationResult], ...]], list[tuple[EvaluationResult, str]]]:
    """Rank inside each Target Task and keep only the top models worth deploying.

    A model that did not beat the mean-prediction baseline carries no deployable
    value, so the R2 floor is applied to every rank including the first one.  The
    second element of the return value records why each dropped result was
    rejected so the run log and the report can stay auditable.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    retained: dict[str, tuple[tuple[int, EvaluationResult], ...]] = {}
    rejected: list[tuple[EvaluationResult, str]] = []
    for task_key, ranked in rank_within_target_tasks(results).items():
        kept: list[tuple[int, EvaluationResult]] = []
        for rank, result in enumerate(ranked, start=1):
            reported = result.metrics.reported_r2
            if not np.isfinite(reported) or reported <= minimum_reported_r2:
                rejected.append(
                    (
                        result,
                        f"rank {rank}: Reported R2={reported:.4f} is not above "
                        f"{minimum_reported_r2:.2f}; no better than the mean baseline",
                    )
                )
                continue
            if len(kept) >= limit:
                rejected.append((result, f"rank {rank}: outside the top {limit} of this Target Task"))
                continue
            kept.append((len(kept) + 1, result))
        if kept:
            retained[task_key] = tuple(kept)
    return retained, rejected


def bundle_metadata(
    bundle: FittedModelBundle,
    *,
    rank: int | None = None,
    evidence_stage: str = "final_fit_only",
    metrics: RegressionMetrics | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sidecar metadata so the model library and the inference UI need no dataset.

    Feature and target names are recorded at training time, which is what lets
    the inference UI build its input fields and result cards from the model
    alone.
    """

    experiment = bundle.experiment
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "regression_v4_bundle",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "rank": rank,
        "evidence_stage": evidence_stage,
        "target_task": experiment.target_task.key,
        "target_task_name": experiment.target_task.display_name,
        "experiment_key": experiment.stable_key,
        "experiment": experiment.display_name,
        "model_id": bundle.model_id,
        "model_name": bundle.model_name,
        "loss": bundle.loss_id,
        "scaler": bundle.scaler_id,
        "augmentation": bundle.augmentation_id,
        "repetition": bundle.repetition,
        "model_parameters": dict(bundle.model_parameters or {}),
        "configuration": model_configuration(bundle.model_id),
        "feature_names": list(experiment.input_columns),
        "feature_labels": list(experiment.input_labels or experiment.input_columns),
        "target_names": list(experiment.output_columns),
        "target_labels": list(experiment.output_labels or experiment.output_columns),
        "feature_structure": experiment.feature_structure.value,
        "feature_groups": list(experiment.feature_groups),
        "training_rows": bundle.training_rows,
        "trained_epochs": bundle.trained_epochs,
        "evidence_note": bundle.evidence_note,
        "metrics": metrics.to_dict() if metrics is not None else None,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def save_fitted_models(
    bundles: Sequence[FittedModelBundle],
    output_folder: str | Path,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> tuple[Path, ...]:
    import joblib

    if metadata is not None and len(metadata) != len(bundles):
        raise ValueError("metadata must align with bundles")
    output = Path(output_folder).expanduser().resolve() / "models"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, bundle in enumerate(bundles):
        path = output / (
            f"{bundle.experiment.stable_key}__{bundle.model_id}__{bundle.loss_id}__"
            f"{bundle.scaler_id}__{bundle.augmentation_id.replace(':', '_').replace('+', '_')}__rep{bundle.repetition}.joblib"
        )
        joblib.dump(bundle, path)
        entry = (metadata[index] if metadata is not None else None) or bundle_metadata(bundle)
        entry = dict(entry)
        entry.setdefault("bundle_file", path.name)
        entry["run_folder"] = str(Path(output_folder).expanduser().resolve())
        path.with_suffix(".meta.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        paths.append(path)
    return tuple(paths)


def rank_within_target_tasks(results: Sequence[EvaluationResult]) -> dict[str, tuple[EvaluationResult, ...]]:
    """Rank by macro NMAE only within an identical output-column task."""

    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.target_task_key, []).append(result)
    return {
        task: tuple(
            sorted(rows, key=lambda row: (row.metrics.nmae, row.metrics.mae, row.metrics.rmse, row.model_name))
        )
        for task, rows in grouped.items()
    }
