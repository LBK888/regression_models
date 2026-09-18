"""Regression metrics with separate diagnostic and presentation R² values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    nmae: float
    mape_percent: float
    diagnostic_r2: float
    reported_r2: float
    per_target: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "nmae": self.nmae,
            "mape_percent": self.mape_percent,
            "diagnostic_r2": self.diagnostic_r2,
            "reported_r2": self.reported_r2,
            "per_target": [dict(row) for row in self.per_target],
        }


def _as_2d(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")
    return array


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _target_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = np.maximum(np.abs(y_true), np.finfo(float).eps)
    mape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)
    diagnostic = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan")
    reported = max(0.0, diagnostic) if np.isfinite(diagnostic) else diagnostic
    nmae = mae / max(float(np.mean(np.abs(y_true))), np.finfo(float).eps)
    return {
        "target": name,
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "mape_percent": mape,
        "diagnostic_r2": diagnostic,
        "reported_r2": reported,
    }


def calculate_regression_metrics(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    target_names: Sequence[str] | None = None,
) -> RegressionMetrics:
    """Calculate metrics without hiding negative model diagnostics.

    ``reported_r2`` is clamped to zero for user-facing summaries.  The signed
    ``diagnostic_r2`` is retained for machine-readable output and appendices.
    """

    truth = _as_2d(y_true)
    prediction = _as_2d(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError(f"y_true/y_pred shapes differ: {truth.shape} != {prediction.shape}")
    names = tuple(target_names or (f"target_{index + 1}" for index in range(truth.shape[1])))
    if len(names) != truth.shape[1]:
        raise ValueError("target_names length must equal the number of output columns")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite y_true and y_pred values")

    rows = tuple(
        _target_metrics(truth[:, index], prediction[:, index], str(names[index]))
        for index in range(truth.shape[1])
    )
    return RegressionMetrics(
        mae=_finite_mean([row["mae"] for row in rows]),
        rmse=_finite_mean([row["rmse"] for row in rows]),
        nmae=_finite_mean([row["nmae"] for row in rows]),
        mape_percent=_finite_mean([row["mape_percent"] for row in rows]),
        diagnostic_r2=_finite_mean([row["diagnostic_r2"] for row in rows]),
        reported_r2=_finite_mean([row["reported_r2"] for row in rows]),
        per_target=rows,
    )
