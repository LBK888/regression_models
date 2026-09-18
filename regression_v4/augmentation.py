"""Training-only augmentation algorithms with explicit statistical semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError(f"Expected aligned 2D X/y, got {features.shape} and {targets.shape}")
    if features.shape[0] < 2:
        raise ValueError("Augmentation requires at least two training observations")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Augmentation requires finite values")
    return features, targets


def c_mixup(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    random_seed: int,
    alpha: float = 2.0,
    target_bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-aware interpolation using C-Mixup-style neighbor probabilities."""

    features, targets = _validate_xy(X, y)
    if count < 0 or alpha <= 0:
        raise ValueError("count must be nonnegative and alpha must be positive")
    if count == 0:
        return features[:0].copy(), targets[:0].copy()
    rng = np.random.default_rng(random_seed)
    target_scale = np.std(targets, axis=0, ddof=1)
    target_scale[target_scale <= np.finfo(float).eps] = 1.0
    normalized = targets / target_scale
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    positive = distances[distances > 0]
    bandwidth = float(
        target_bandwidth
        if target_bandwidth is not None
        else (np.median(positive) if positive.size else 1.0)
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("target_bandwidth must be finite and positive")
    bandwidth = max(bandwidth, np.finfo(float).eps)

    # Stable row-wise softmax.  Direct exponentiation can underflow an entire
    # outlier row to zero; subtracting the best non-self log-weight preserves
    # its closest target neighbor instead of falling back to arbitrary pairing.
    with np.errstate(over="ignore", invalid="ignore"):
        logits = -0.5 * np.square(distances / bandwidth)
    np.fill_diagonal(logits, -np.inf)
    row_max = np.max(logits, axis=1, keepdims=True)
    valid_rows = np.isfinite(row_max[:, 0])
    weights = np.zeros_like(logits)
    weights[valid_rows] = np.exp(logits[valid_rows] - row_max[valid_rows])

    # A finite input can still overflow after squaring at truly extreme scale.
    # In that numerical corner, use the closest non-self target deterministically.
    if not np.all(valid_rows):
        nonself_distances = distances.copy()
        np.fill_diagonal(nonself_distances, np.inf)
        fallback_rows = np.flatnonzero(~valid_rows)
        nearest = np.argmin(nonself_distances[fallback_rows], axis=1)
        weights[fallback_rows, nearest] = 1.0

    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
        raise RuntimeError("C-Mixup could not construct valid target-neighbor probabilities")
    weights /= row_sums
    # Correct the last few ulps on the largest entry so strict RNG backends see
    # a probability mass of exactly one within floating-point arithmetic.
    largest = np.argmax(weights, axis=1)
    weights[np.arange(len(weights)), largest] += 1.0 - weights.sum(axis=1)
    anchors = rng.integers(0, len(features), size=count)
    partners = np.asarray([rng.choice(len(features), p=weights[index]) for index in anchors])
    mixing = rng.beta(alpha, alpha, size=(count, 1))
    return (
        mixing * features[anchors] + (1.0 - mixing) * features[partners],
        mixing * targets[anchors] + (1.0 - mixing) * targets[partners],
    )


def calibrated_feature_noise(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    random_seed: int,
    noise_fraction: float = 0.02,
    replicate_sd: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb X only, calibrated by replicate SD or training-feature dispersion."""

    features, targets = _validate_xy(X, y)
    if count < 0 or noise_fraction < 0:
        raise ValueError("count and noise_fraction must be nonnegative")
    rng = np.random.default_rng(random_seed)
    indexes = rng.integers(0, len(features), size=count)
    if replicate_sd is None:
        scale = np.std(features, axis=0, ddof=1) * noise_fraction
    else:
        scale = np.asarray(replicate_sd, dtype=float)
        if scale.shape != (features.shape[1],):
            raise ValueError("replicate_sd must have one value per feature")
    noisy = features[indexes] + rng.normal(0.0, scale, size=(count, features.shape[1]))
    return noisy, targets[indexes].copy()


def foma_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    random_seed: int,
    alpha: float = 2.0,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate FOMA samples by shrinking non-leading joint X/y SVD modes.

    This is an exact-row-count, NumPy implementation of the Simple FOMA
    transform used by FOMA_lite.  The SVD is learned only from the current
    fold's scaled training partition.  Each synthetic observation uses a
    Beta(alpha, alpha) shrink factor for singular components after the first
    ``k`` components.
    """

    features, targets = _validate_xy(X, y)
    if count < 0 or alpha <= 0 or k < 1:
        raise ValueError("count must be nonnegative, alpha positive, and k at least 1")
    if count == 0:
        return features[:0].copy(), targets[:0].copy()

    joint = np.concatenate([features, targets], axis=1)
    left, singular_values, right_t = np.linalg.svd(joint, full_matrices=False)
    rank = len(singular_values)
    if k >= rank:
        raise ValueError(f"FOMA k must be smaller than the joint X/y rank ({rank})")

    retained = (left[:, :k] * singular_values[:k]) @ right_t[:k]
    remainder = (left[:, k:] * singular_values[k:]) @ right_t[k:]
    rng = np.random.default_rng(random_seed)
    source_indexes = rng.integers(0, len(joint), size=count)
    shrink = rng.beta(alpha, alpha, size=(count, 1))
    synthetic = retained[source_indexes] + shrink * remainder[source_indexes]
    return synthetic[:, : features.shape[1]], synthetic[:, features.shape[1] :]


def spectral_perturbation(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    random_seed: int,
    axis: Sequence[float],
    additive_fraction: float = 0.01,
    multiplicative_fraction: float = 0.01,
    baseline_fraction: float = 0.01,
    source_indexes: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply smooth additive, multiplicative and baseline drift to spectra."""

    features, targets = _validate_xy(X, y)
    positions = np.asarray(axis, dtype=float)
    if positions.shape != (features.shape[1],) or not np.all(np.diff(positions) > 0):
        raise ValueError("axis must be strictly increasing with one coordinate per feature")
    rng = np.random.default_rng(random_seed)
    if source_indexes is None:
        indexes = rng.integers(0, len(features), size=count)
    else:
        indexes = np.asarray(source_indexes, dtype=int)
        if indexes.shape != (count,) or np.any(indexes < 0) or np.any(indexes >= len(features)):
            raise ValueError("source_indexes must contain one valid training-row index per synthetic row")
    base = features[indexes].copy()
    feature_sd = np.std(features, axis=0, ddof=1)
    scale = float(np.nanmedian(feature_sd[feature_sd > 0])) if np.any(feature_sd > 0) else 1.0
    axis01 = (positions - positions.min()) / max(np.ptp(positions), np.finfo(float).eps)
    multiplicative = rng.normal(1.0, multiplicative_fraction, size=(count, 1))
    offset = rng.normal(0.0, baseline_fraction * scale, size=(count, 1))
    slope = rng.normal(0.0, baseline_fraction * scale, size=(count, 1))
    additive = rng.normal(0.0, additive_fraction * scale, size=base.shape)
    augmented = base * multiplicative + offset + slope * (axis01.reshape(1, -1) - 0.5) + additive
    return augmented, targets[indexes].copy()


@dataclass(frozen=True)
class FeatureMaskResult:
    features: np.ndarray
    mask: np.ndarray


def stochastic_feature_mask(
    scaled_X: np.ndarray,
    *,
    random_seed: int,
    probability: float = 0.05,
    add_indicators: bool = True,
) -> FeatureMaskResult:
    """Training-time per-observation masking, applied after feature scaling."""

    features = np.asarray(scaled_X, dtype=float)
    if features.ndim != 2 or not 0 <= probability < 1:
        raise ValueError("scaled_X must be 2D and probability must be in [0, 1)")
    rng = np.random.default_rng(random_seed)
    mask = rng.random(features.shape) < probability
    masked = features.copy()
    masked[mask] = 0.0  # Scaled training mean, not an arbitrary raw-space zero.
    output = np.concatenate([masked, mask.astype(float)], axis=1) if add_indicators else masked
    return FeatureMaskResult(output, mask)
