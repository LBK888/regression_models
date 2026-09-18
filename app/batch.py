# -*- coding: utf-8 -*-
"""Batch inference over a whole worksheet, with optional test-data validation.

The single-row inference form answers "what does this model say about this
sample". This module answers the two questions that need a table:

* run one or more models over every row of a sheet and write the predictions
  back out, and
* when the sheet also carries the true target values, treat it as test data and
  measure the models against it.

Metrics come from :mod:`regression_v4.metrics` so a validation number produced
here means exactly what the same-named number in a Benchmark Run report means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from i18n import tr
from regression_core import DatasetBundle, FlexibleDatasetLoader
from regression_v4.metrics import RegressionMetrics, calculate_regression_metrics

from .adapters import Predictor


def load_table(path: str | Path) -> DatasetBundle:
    """Load a CSV/XLSX with the same auditing loader the training tab uses."""

    return FlexibleDatasetLoader().load(path)


def sheet_frame(bundle: DatasetBundle, sheet: str) -> pd.DataFrame:
    return bundle.sheets[sheet].data


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Comparison key that ignores case, spacing and punctuation differences."""

    tail = str(name).rsplit("::", 1)[-1]
    return re.sub(r"[^0-9a-z一-鿿]+", "", tail.lower())


def suggest_mapping(
    names: Sequence[str], columns: Sequence[str]
) -> dict[str, str | None]:
    """Match model feature/target names to sheet columns.

    Tries an exact match first, then the ``Sheet::Column`` tail, then a
    normalized form. Names that match nothing come back as ``None`` so the UI
    can ask the user instead of guessing.
    """

    available = list(dict.fromkeys(str(column) for column in columns))
    exact = {column: column for column in available}
    normalized: dict[str, str] = {}
    for column in available:
        normalized.setdefault(_normalize(column), column)

    mapping: dict[str, str | None] = {}
    taken: set[str] = set()
    for name in names:
        text = str(name)
        candidate = exact.get(text)
        if candidate is None:
            candidate = exact.get(text.rsplit("::", 1)[-1])
        if candidate is None:
            candidate = normalized.get(_normalize(text))
        if candidate is not None and candidate in taken:
            candidate = None
        if candidate is not None:
            taken.add(candidate)
        mapping[text] = candidate
    return mapping


# ---------------------------------------------------------------------------
# Running a batch
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Predictions for one model over one sheet, plus validation if possible."""

    model_label: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    row_labels: tuple[str, ...]
    predictions: np.ndarray
    ground_truth: np.ndarray | None = None
    metrics: RegressionMetrics | None = None
    used_rows: int = 0
    total_rows: int = 0
    dropped_missing_features: int = 0
    dropped_missing_targets: int = 0
    error: str = ""
    notes: tuple[str, ...] = ()

    @property
    def has_validation(self) -> bool:
        return self.metrics is not None and self.ground_truth is not None

    def predictions_frame(self) -> pd.DataFrame:
        """Long-form predictions, one row per observation and target."""

        rows: list[dict[str, Any]] = []
        for row_index, label in enumerate(self.row_labels):
            for target_index, target in enumerate(self.target_names):
                entry: dict[str, Any] = {
                    "model": self.model_label,
                    "observation_id": label,
                    "target": target,
                    "predicted": float(self.predictions[row_index, target_index]),
                }
                if self.ground_truth is not None:
                    actual = float(self.ground_truth[row_index, target_index])
                    entry["actual"] = actual
                    entry["residual"] = actual - entry["predicted"]
                    denominator = max(abs(actual), np.finfo(float).eps)
                    entry["absolute_percentage_error"] = abs(entry["residual"]) / denominator * 100.0
                rows.append(entry)
        return pd.DataFrame(rows)

    def wide_frame(self) -> pd.DataFrame:
        """One row per observation, ready to paste back into a worksheet."""

        data: dict[str, Any] = {"observation_id": list(self.row_labels)}
        for index, target in enumerate(self.target_names):
            data[f"{target} (predicted)"] = self.predictions[:, index]
            if self.ground_truth is not None:
                data[f"{target} (actual)"] = self.ground_truth[:, index]
                data[f"{target} (residual)"] = self.ground_truth[:, index] - self.predictions[:, index]
        return pd.DataFrame(data)

    def metric_rows(self) -> list[dict[str, Any]]:
        """Macro plus per-target metric rows for the comparison table."""

        if self.metrics is None:
            return []
        rows = [
            {
                "model": self.model_label,
                "target": "ALL (macro)",
                "n": self.used_rows,
                "mae": self.metrics.mae,
                "rmse": self.metrics.rmse,
                "nmae": self.metrics.nmae,
                "mape_percent": self.metrics.mape_percent,
                "reported_r2": self.metrics.reported_r2,
                "diagnostic_r2": self.metrics.diagnostic_r2,
            }
        ]
        for entry in self.metrics.per_target:
            rows.append(
                {
                    "model": self.model_label,
                    "target": entry["target"],
                    "n": self.used_rows,
                    "mae": entry["mae"],
                    "rmse": entry["rmse"],
                    "nmae": entry["nmae"],
                    "mape_percent": entry["mape_percent"],
                    "reported_r2": entry["reported_r2"],
                    "diagnostic_r2": entry["diagnostic_r2"],
                }
            )
        return rows


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    data = [pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) for column in columns]
    return np.column_stack(data) if data else np.empty((len(frame), 0))


def run_batch(
    predictor: Predictor,
    frame: pd.DataFrame,
    feature_mapping: Mapping[str, str],
    *,
    model_label: str,
    target_mapping: Mapping[str, str] | None = None,
    observation_column: str | None = None,
) -> BatchResult:
    """Predict every usable row of ``frame`` and validate against ground truth.

    Rows with a missing or non-numeric value in any mapped feature cannot be
    predicted and are dropped as one cohort, the same rule the training loader
    applies, and the count is reported rather than silently absorbed.
    """

    feature_names = tuple(predictor.feature_names)
    target_names = tuple(predictor.target_names)
    missing = [name for name in feature_names if not feature_mapping.get(name)]
    if missing:
        return BatchResult(
            model_label=model_label,
            feature_names=feature_names,
            target_names=target_names,
            row_labels=(),
            predictions=np.empty((0, len(target_names))),
            total_rows=len(frame),
            error=tr(
                "{count} input column(s) are still unmapped: {names}",
                count=len(missing),
                names=", ".join(missing[:6]) + ("…" if len(missing) > 6 else ""),
            ),
        )

    feature_columns = [str(feature_mapping[name]) for name in feature_names]
    absent = [column for column in feature_columns if column not in frame.columns]
    if absent:
        return BatchResult(
            model_label=model_label,
            feature_names=feature_names,
            target_names=target_names,
            row_labels=(),
            predictions=np.empty((0, len(target_names))),
            total_rows=len(frame),
            error=tr(
                "The table has no such column(s): {names}", names=", ".join(absent[:6])
            ),
        )

    features = _numeric_matrix(frame, feature_columns)
    valid = np.isfinite(features).all(axis=1)
    dropped_features = int((~valid).sum())

    truth: np.ndarray | None = None
    dropped_targets = 0
    notes: list[str] = []
    resolved_targets = {
        name: str(column)
        for name, column in (target_mapping or {}).items()
        if column and str(column) in frame.columns
    }
    if resolved_targets and len(resolved_targets) == len(target_names):
        truth_all = _numeric_matrix(frame, [resolved_targets[name] for name in target_names])
        truth_valid = np.isfinite(truth_all).all(axis=1)
        dropped_targets = int((valid & ~truth_valid).sum())
        valid = valid & truth_valid
        truth = truth_all
    elif resolved_targets:
        notes.append(
            tr(
                "Only some ground-truth columns are mapped, so this run produces "
                "predictions without validation."
            )
        )

    if not valid.any():
        return BatchResult(
            model_label=model_label,
            feature_names=feature_names,
            target_names=target_names,
            row_labels=(),
            predictions=np.empty((0, len(target_names))),
            total_rows=len(frame),
            dropped_missing_features=dropped_features,
            dropped_missing_targets=dropped_targets,
            error=tr(
                "No row has a complete set of input values (and ground truth, "
                "where it was mapped)."
            ),
        )

    usable = frame.loc[valid]
    if observation_column and observation_column in frame.columns:
        labels = tuple(str(value) for value in usable[observation_column].tolist())
    else:
        labels = tuple(f"row::{index}" for index in usable.index.tolist())

    try:
        predictions = predictor.predict(features[valid])
    except Exception as exc:
        return BatchResult(
            model_label=model_label,
            feature_names=feature_names,
            target_names=target_names,
            row_labels=(),
            predictions=np.empty((0, len(target_names))),
            total_rows=len(frame),
            dropped_missing_features=dropped_features,
            dropped_missing_targets=dropped_targets,
            error=tr(
                "Inference failed: {error_type}: {error}",
                error_type=type(exc).__name__,
                error=str(exc),
            ),
        )

    metrics: RegressionMetrics | None = None
    selected_truth = truth[valid] if truth is not None else None
    if selected_truth is not None:
        if len(selected_truth) < 2:
            notes.append(
                tr("Validation needs at least two ground-truth rows; R² cannot be computed.")
            )
        else:
            try:
                metrics = calculate_regression_metrics(selected_truth, predictions, target_names)
            except Exception as exc:
                notes.append(tr("Metric calculation failed: {error}", error=str(exc)))

    if dropped_features:
        notes.append(
            tr(
                "{count} row(s) were skipped because an input value was missing or "
                "not numeric.",
                count=dropped_features,
            )
        )
    if dropped_targets:
        notes.append(
            tr(
                "{count} row(s) were left out of the validation because ground truth "
                "was missing.",
                count=dropped_targets,
            )
        )

    return BatchResult(
        model_label=model_label,
        feature_names=feature_names,
        target_names=target_names,
        row_labels=labels,
        predictions=predictions,
        ground_truth=selected_truth,
        metrics=metrics,
        used_rows=int(valid.sum()),
        total_rows=len(frame),
        dropped_missing_features=dropped_features,
        dropped_missing_targets=dropped_targets,
        notes=tuple(notes),
    )


def rank_results(results: Sequence[BatchResult]) -> list[BatchResult]:
    """Rank validated models by macro NMAE, the same rule a Benchmark Run uses."""

    validated = [result for result in results if result.has_validation]
    return sorted(
        validated,
        key=lambda result: (
            result.metrics.nmae,
            result.metrics.mae,
            result.metrics.rmse,
            result.model_label,
        ),
    )
