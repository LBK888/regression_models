# -*- coding: utf-8 -*-
"""One prediction interface for every model kind the library can hold.

Three generations of artifact have to work side by side:

``regression_v4`` bundle (``.joblib``)
    A pickled :class:`regression_v4.evaluation.FittedModelBundle`.  It carries
    the estimator, both scalers and the Experiment identity, so every model the
    benchmark can train — linear, kernel, tree ensemble, gradient boosting,
    legacy MLP, CNN1D/ResNet1D, Grouped Fusion, embedding MLP, FT-Transformer,
    ModernNCA, TabM, RealMLP — reaches inference through the same
    ``bundle.predict`` path.

legacy package (``.pkl`` + optional ``.md``)
    The v2.2 sensor format: a dict of ``model_state_dict``, ``scaler``,
    ``feature_names``, ``target_names``.  The architecture is rediscovered by
    trying candidate classes until one accepts the state dict.

third-party estimator (``.joblib`` / ``.pkl``)
    Anything exposing ``predict``.  Feature and target names are taken from
    scikit-learn attributes or a sidecar file, and the entry stays usable even
    when only the feature count is known.

Anything that cannot be turned into a predictor is reported with a reason
instead of being hidden, so an incompatible file is visible and explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.machinery
import importlib.util
import io
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

import legacy_architectures
import regression_core
from i18n import tr


MODEL_SUFFIXES = (".joblib", ".pkl", ".pickle", ".pt", ".pth")

_SIDECAR_SUFFIX = ".meta.json"


# ---------------------------------------------------------------------------
# Architecture candidates for legacy state-dict packages
# ---------------------------------------------------------------------------

class Wide_Network(nn.Module):
    """Wide MLP 512-256-128 with BatchNorm, as used by early sensor models."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Wide_Network_NoBN(nn.Module):
    """Wide MLP variant without BatchNorm."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Deep_Network(nn.Module):
    """Deep MLP, 256 wide for four layers."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Deep_Network_BN(nn.Module):
    """Deep MLP with BatchNorm."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Standard_Network(nn.Module):
    """Standard MLP 128-64-32."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Narrow_Network(nn.Module):
    """Narrow MLP 64-32."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


#: Preferred class per recorded architecture name, then its known variants.
ARCHITECTURE_VARIANTS: dict[str, tuple[type, ...]] = {
    "Wide_Network": (regression_core.WideNetwork, Wide_Network, Wide_Network_NoBN),
    "WideNetwork": (regression_core.WideNetwork, Wide_Network, Wide_Network_NoBN),
    "Deep_Network": (Deep_Network, Deep_Network_BN, regression_core.DeepFFN),
    "Deep_FFN": (regression_core.DeepFFN, legacy_architectures.DeepFFN, Deep_Network),
    "DeepFFN": (regression_core.DeepFFN, legacy_architectures.DeepFFN),
    "Standard_Network": (Standard_Network,),
    "Narrow_Network": (Narrow_Network,),
    "ResNet": (regression_core.ResNet, legacy_architectures.ResNet),
    "AutoEncoder_Net": (regression_core.AutoEncoderNet, legacy_architectures.AutoEncoderNet),
    "AutoEncoderNet": (regression_core.AutoEncoderNet, legacy_architectures.AutoEncoderNet),
    "Ensemble_Net": (regression_core.EnsembleNet, legacy_architectures.EnsembleNet),
    "EnsembleNet": (regression_core.EnsembleNet, legacy_architectures.EnsembleNet),
}

#: Fallback sweep when the recorded name is unknown or absent.
ALL_ARCHITECTURES: tuple[type, ...] = (
    regression_core.DeepFFN,
    regression_core.WideNetwork,
    regression_core.ResNet,
    regression_core.EnsembleNet,
    regression_core.AutoEncoderNet,
    Wide_Network,
    Wide_Network_NoBN,
    Deep_Network,
    Deep_Network_BN,
    Standard_Network,
    Narrow_Network,
    legacy_architectures.DeepFFN,
    legacy_architectures.WideNetwork,
    legacy_architectures.ResNet,
    legacy_architectures.EnsembleNet,
    legacy_architectures.AutoEncoderNet,
)


class _CPUUnpickler(pickle.Unpickler):
    """Load CUDA-saved tensors on a CPU-only machine."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        if torch.cuda.is_available():
            try:
                return pickle.load(handle)
            except Exception:
                handle.seek(0)
        return _CPUUnpickler(handle).load()


# ---------------------------------------------------------------------------
# Sidecar metadata
# ---------------------------------------------------------------------------

def parse_markdown_sidecar(path: Path) -> dict[str, Any]:
    """Read the v2.2 ``.md`` usage note that ships next to a legacy model.

    The legacy packages do not always store feature/target names inside the
    pickle, so the names recorded in the generated Markdown are the backfill.
    """

    info: dict[str, Any] = {
        "model_name": path.stem,
        "architecture": "",
        "augmentation": "",
        "scaler": "",
        "feature_names": [],
        "target_names": [],
        "metrics": {},
    }
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info

    for key, pattern in (
        ("model_name", r"\*\*Model Name\*\*:\s*(.+)"),
        ("architecture", r"\*\*Architecture\*\*:\s*(.+)"),
        ("augmentation", r"\*\*Augmentation\*\*:\s*(.+)"),
        ("scaler", r"\*\*Scaler\*\*:\s*(.+)"),
    ):
        match = re.search(pattern, content)
        if match:
            info[key] = match.group(1).strip()

    for key, pattern in (
        ("feature_names", r"\*\*Features\*\*:\s*(\[.+?\])"),
        ("target_names", r"\*\*Targets\*\*:\s*(\[.+?\])"),
    ):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            info[key] = re.findall(r"'([^']+)'", match.group(1))

    metrics: dict[str, float] = {}
    for key, pattern in (
        ("mae", r"MAE:\s*([\d.eE+-]+)"),
        ("rmse", r"RMSE:\s*([\d.eE+-]+)"),
        ("reported_r2", r"R[²2]:\s*([-\d.eE+]+)"),
        ("trained_epochs", r"Stopped at Epoch:\s*(\d+)"),
    ):
        match = re.search(pattern, content)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                continue
    if "reported_r2" in metrics:
        metrics.setdefault("diagnostic_r2", metrics["reported_r2"])
    info["metrics"] = metrics
    return info


def read_sidecar(model_path: Path) -> dict[str, Any]:
    """Merge every metadata source that sits next to a model file."""

    merged: dict[str, Any] = {}
    json_path = model_path.with_suffix(_SIDECAR_SUFFIX)
    if not json_path.exists():
        json_path = model_path.with_name(model_path.stem + _SIDECAR_SUFFIX)
    if json_path.exists():
        try:
            merged.update(json.loads(json_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    markdown_path = model_path.with_suffix(".md")
    if markdown_path.exists():
        note = parse_markdown_sidecar(markdown_path)
        merged.setdefault("model_name", note["model_name"])
        merged.setdefault("architecture", note["architecture"])
        merged.setdefault("scaler", note["scaler"])
        merged.setdefault("augmentation", note["augmentation"])
        if note["feature_names"]:
            merged.setdefault("feature_names", note["feature_names"])
        if note["target_names"]:
            merged.setdefault("target_names", note["target_names"])
        if note["metrics"] and not merged.get("metrics"):
            merged["metrics"] = note["metrics"]
    return merged


def _external_architectures(folder: Path) -> dict[str, type]:
    """Load user-supplied ``architectures.py`` beside a model, if present."""

    path = folder / "architectures.py"
    if not path.exists():
        return {}
    try:
        loader = importlib.machinery.SourceFileLoader(f"_user_arch_{abs(hash(str(path)))}", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = module
        spec.loader.exec_module(module)
    except Exception:
        return {}
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    }


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

@dataclass
class Predictor:
    """A loaded model reduced to names plus a matrix-in, matrix-out call."""

    kind: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    architecture: str = ""
    device: str = "cpu"
    notes: tuple[str, ...] = ()
    _call: Any = field(default=None, repr=False)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Predict a ``(n_rows, n_features)`` matrix as ``(n_rows, n_targets)``."""

        features = np.asarray(matrix, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                tr(
                    "Expected {expected} features, received {received}",
                    expected=len(self.feature_names),
                    received=features.shape[1],
                )
            )
        prediction = np.asarray(self._call(features), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if prediction.shape[0] != features.shape[0]:
            raise ValueError(
                tr("The model returned a different number of rows than it was given")
            )
        expected = len(self.target_names)
        if prediction.shape[1] != expected:
            raise ValueError(
                tr(
                    "The model returned {returned} output column(s) but {expected} "
                    "target name(s) are recorded",
                    returned=prediction.shape[1],
                    expected=expected,
                )
            )
        return prediction


class ModelLoadError(RuntimeError):
    """The file exists but cannot be turned into a usable predictor."""


def _names(values: Sequence[Any] | None, count: int, prefix: str) -> tuple[str, ...]:
    cleaned = [str(value) for value in (values or []) if str(value).strip()]
    if len(cleaned) == count:
        return tuple(cleaned)
    return tuple(f"{prefix}{index + 1}" for index in range(count))


def _is_v4_bundle(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ("experiment", "model", "x_scaler", "y_scaler")) and hasattr(
        obj, "predict"
    )


def _v4_predictor(bundle: Any) -> Predictor:
    experiment = bundle.experiment
    features = tuple(experiment.input_labels or experiment.input_columns)
    targets = tuple(experiment.output_labels or experiment.output_columns)
    estimator = type(getattr(bundle, "model", None)).__name__
    return Predictor(
        kind="v4_bundle",
        feature_names=features,
        target_names=targets,
        architecture=f"{getattr(bundle, 'model_name', estimator)} ({estimator})",
        device=str(regression_core.DEVICE),
        notes=(
            f"loss={getattr(bundle, 'loss_id', '?')}",
            f"scaler={getattr(bundle, 'scaler_id', '?')}",
            f"augmentation={getattr(bundle, 'augmentation_id', '?')}",
        ),
        _call=bundle.predict,
    )


def _legacy_predictor(package: dict[str, Any], path: Path, sidecar: dict[str, Any]) -> Predictor:
    if "model_state_dict" not in package:
        raise ModelLoadError(tr("the pickle has no 'model_state_dict' entry"))
    state_dict = package["model_state_dict"]
    scaler = package.get("scaler")
    feature_names = _names(
        package.get("feature_names") or sidecar.get("feature_names"),
        int(package.get("input_dim") or len(package.get("feature_names") or sidecar.get("feature_names") or ())),
        "feature_",
    )
    target_names = _names(
        package.get("target_names") or sidecar.get("target_names"),
        int(package.get("output_dim") or len(package.get("target_names") or sidecar.get("target_names") or ()) or 1),
        "target_",
    )
    if not feature_names:
        raise ModelLoadError(
            tr("no feature names or input_dim were recorded in the package or its .md")
        )

    input_dim = int(package.get("input_dim") or len(feature_names))
    output_dim = int(package.get("output_dim") or len(target_names))
    architecture_name = str(package.get("architecture") or sidecar.get("architecture") or "")

    candidates: list[tuple[type, str]] = []
    external = _external_architectures(path.parent)
    if architecture_name and architecture_name in external:
        candidates.append((external[architecture_name], f"{architecture_name} (architectures.py)"))
    for cls in ARCHITECTURE_VARIANTS.get(architecture_name, ()):
        candidates.append((cls, cls.__name__))
    for name, cls in external.items():
        candidates.append((cls, f"{name} (architectures.py)"))
    for cls in ALL_ARCHITECTURES:
        candidates.append((cls, cls.__name__))

    device = regression_core.DEVICE
    tried: list[str] = []
    module: nn.Module | None = None
    label = ""
    seen: set[tuple[str, str]] = set()
    for cls, name in candidates:
        key = (cls.__module__, cls.__qualname__)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate = cls(input_dim, output_dim).to(device)
            candidate.load_state_dict(state_dict, strict=True)
        except Exception:
            tried.append(name)
            continue
        module = candidate
        label = name if name == architecture_name else f"{name}"
        break

    if module is None:
        raise ModelLoadError(
            tr(
                "no known architecture accepts this state dict (recorded name "
                "{name}; tried {count} candidate class(es)). Put a matching class in "
                "an architectures.py file next to the model.",
                name=repr(architecture_name or "unknown"),
                count=len(tried),
            )
        )
    module.eval()

    y_scaler = package.get("target_scaler") or package.get("y_scaler")

    def call(features: np.ndarray) -> np.ndarray:
        prepared = scaler.transform(features) if scaler is not None else features
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(prepared, dtype=np.float32)).to(device)
            output = module(tensor).cpu().numpy()
        if y_scaler is not None:
            output = y_scaler.inverse_transform(np.atleast_2d(output))
        return output

    notes = [tr("architecture={name}", name=label)]
    if scaler is not None:
        notes.append(tr("scaler={name}", name=type(scaler).__name__))
    else:
        notes.append(tr("no scaler stored; raw feature values are fed to the network"))
    return Predictor(
        kind="legacy_package",
        feature_names=feature_names,
        target_names=target_names,
        architecture=label,
        device=str(device),
        notes=tuple(notes),
        _call=call,
    )


def _estimator_predictor(estimator: Any, sidecar: dict[str, Any], kind: str) -> Predictor:
    if not hasattr(estimator, "predict"):
        raise ModelLoadError(
            tr(
                "loaded a {type_name} which exposes no predict() method",
                type_name=type(estimator).__name__,
            )
        )
    recorded_features = sidecar.get("feature_labels") or sidecar.get("feature_names")
    attribute_names = getattr(estimator, "feature_names_in_", None)
    if recorded_features:
        feature_count = len(recorded_features)
    elif attribute_names is not None:
        recorded_features = [str(name) for name in attribute_names]
        feature_count = len(recorded_features)
    else:
        feature_count = int(getattr(estimator, "n_features_in_", 0) or 0)
        if feature_count < 1:
            raise ModelLoadError(
                tr(
                    "the number of input features is unknown: the estimator exposes "
                    "neither feature_names_in_ nor n_features_in_, and no sidecar "
                    ".meta.json records them"
                )
            )
    feature_names = _names(recorded_features, feature_count, "feature_")

    recorded_targets = sidecar.get("target_labels") or sidecar.get("target_names")
    target_count = len(recorded_targets) if recorded_targets else 0
    if target_count < 1:
        probe = np.zeros((1, feature_count), dtype=float)
        try:
            probe_output = np.atleast_2d(np.asarray(estimator.predict(probe), dtype=float))
        except Exception as exc:
            raise ModelLoadError(
                tr("a trial prediction failed: {error}", error=str(exc))
            ) from exc
        target_count = int(probe_output.shape[1])
    target_names = _names(recorded_targets, target_count, "target_")

    x_scaler = sidecar.get("_x_scaler")
    y_scaler = sidecar.get("_y_scaler")

    def call(features: np.ndarray) -> np.ndarray:
        prepared = x_scaler.transform(features) if x_scaler is not None else features
        output = np.atleast_2d(np.asarray(estimator.predict(prepared), dtype=float))
        if output.shape[0] == 1 and features.shape[0] != 1:
            output = output.reshape(features.shape[0], -1)
        if y_scaler is not None:
            output = y_scaler.inverse_transform(output)
        return output

    notes: list[str] = []
    if not (sidecar.get("feature_labels") or sidecar.get("feature_names")) and attribute_names is None:
        notes.append(
            tr("feature names were not recorded; placeholder names are shown in order")
        )
    return Predictor(
        kind=kind,
        feature_names=feature_names,
        target_names=target_names,
        architecture=type(estimator).__name__,
        notes=tuple(notes),
        _call=call,
    )


def _wrapped_estimator(payload: dict[str, Any], sidecar: dict[str, Any]) -> Predictor:
    """A dict that wraps an estimator plus optional scalers."""

    estimator = payload.get("model") or payload.get("estimator") or payload.get("regressor")
    if estimator is None:
        raise ModelLoadError(
            tr("the pickled dict has neither 'model_state_dict' nor a 'model'/'estimator' entry")
        )
    merged = dict(sidecar)
    merged.setdefault("feature_names", payload.get("feature_names"))
    merged.setdefault("target_names", payload.get("target_names"))
    merged["_x_scaler"] = payload.get("scaler") or payload.get("x_scaler")
    merged["_y_scaler"] = payload.get("target_scaler") or payload.get("y_scaler")
    return _estimator_predictor(estimator, merged, "wrapped_estimator")


def load_predictor(path: str | Path) -> Predictor:
    """Load any supported model file and return a uniform predictor.

    Raises :class:`ModelLoadError` with a human-readable reason when the file
    cannot be used, which the model library shows in its compatibility column.
    """

    model_path = Path(path)
    if not model_path.exists():
        raise ModelLoadError(tr("the file no longer exists"))
    sidecar = read_sidecar(model_path)
    suffix = model_path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelLoadError(tr("torch.load failed: {error}", error=str(exc))) from exc
        if isinstance(payload, nn.Module):
            raise ModelLoadError(
                tr(
                    "a bare nn.Module was saved without feature/target names; save a "
                    "regression_v4 bundle or add a sidecar .meta.json"
                )
            )
        if isinstance(payload, dict) and "model_state_dict" in payload:
            return _legacy_predictor(payload, model_path, sidecar)
        if isinstance(payload, dict):
            return _legacy_predictor({"model_state_dict": payload, **sidecar}, model_path, sidecar)
        raise ModelLoadError(
            tr("unsupported .pt payload of type {type_name}", type_name=type(payload).__name__)
        )

    if suffix == ".joblib":
        try:
            import joblib

            payload = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(tr("joblib.load failed: {error}", error=str(exc))) from exc
    elif suffix in {".pkl", ".pickle"}:
        try:
            payload = _load_pickle(model_path)
        except Exception as exc:
            raise ModelLoadError(tr("pickle.load failed: {error}", error=str(exc))) from exc
    else:
        raise ModelLoadError(tr("unsupported file type {suffix}", suffix=repr(suffix)))

    if _is_v4_bundle(payload):
        return _v4_predictor(payload)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return _legacy_predictor(payload, model_path, sidecar)
        return _wrapped_estimator(payload, sidecar)
    return _estimator_predictor(payload, sidecar, "third_party_estimator")


KIND_LABELS = {
    "v4_bundle": "regression_v4 bundle",
    "legacy_package": "legacy state-dict package",
    "wrapped_estimator": "wrapped estimator",
    "third_party_estimator": "third-party estimator",
}
