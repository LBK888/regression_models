# -*- coding: utf-8 -*-
"""The model library: what exists, what runs, and what the user wants to see.

Training is run many times with different purposes, so listing every saved
bundle in the inference tab would be unusable.  The library is the middle
layer: it discovers models under ``models/`` and under every Benchmark Run,
groups them by run, records a user-owned name and an "use in inference" flag,
and reports each model's Reported R² and MAPE so the user can decide which ones
are worth keeping visible.

The user-owned part (names, flags, notes) lives in ``models/library.json`` and
is keyed by a path relative to the project root, so renaming a model in the UI
never touches the model file itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from i18n import tr
from project_paths import (
    DEFAULT_OUTPUT_ROOT,
    LIBRARY_INDEX_PATH,
    MODEL_LIBRARY_ROOT,
    PROJECT_ROOT,
    model_search_roots,
)

from .adapters import KIND_LABELS, MODEL_SUFFIXES, ModelLoadError, Predictor, load_predictor, read_sidecar


METRIC_KEYS = ("mae", "rmse", "nmae", "mape_percent", "reported_r2", "diagnostic_r2")


@dataclass
class ModelEntry:
    """One discovered model file, its metadata and its user-owned settings."""

    entry_id: str
    path: Path
    run_label: str
    run_folder: str
    default_name: str
    display_name: str
    enabled: bool
    notes: str
    kind: str = "unknown"
    model_id: str = ""
    model_name: str = ""
    experiment: str = ""
    target_task: str = ""
    rank: int | None = None
    evidence_stage: str = ""
    architecture: str = ""
    feature_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    size_bytes: int = 0
    compatible: bool = False
    reason: str = "not probed yet"
    probed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def kind_label(self) -> str:
        return tr(KIND_LABELS.get(self.kind, self.kind or "unknown"))

    @property
    def reported_r2(self) -> float | None:
        value = self.metrics.get("reported_r2")
        return float(value) if value is not None else None

    @property
    def mape_percent(self) -> float | None:
        value = self.metrics.get("mape_percent")
        return float(value) if value is not None else None

    @property
    def usable(self) -> bool:
        return self.compatible and self.enabled

    def summary_line(self) -> str:
        parts = [self.display_name]
        if self.architecture:
            parts.append(self.architecture)
        r2 = self.reported_r2
        if r2 is not None:
            parts.append(f"R²={r2:.4f}")
        mape = self.mape_percent
        if mape is not None:
            parts.append(f"MAPE={mape:.2f}%")
        parts.append(f"{len(self.feature_names)} in / {len(self.target_names)} out")
        return " · ".join(parts)


def _relative_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _run_label_for(path: Path, meta: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(run label, run folder)`` for grouping in the library table."""

    recorded = str(meta.get("run_label") or "").strip()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(DEFAULT_OUTPUT_ROOT.resolve())
    except ValueError:
        try:
            relative = resolved.relative_to(MODEL_LIBRARY_ROOT.resolve())
        except ValueError:
            return (recorded or tr("External path"), str(resolved.parent))
        folder = relative.parent.as_posix()
        label = recorded or (
            tr("Model library / {folder}", folder=folder)
            if folder not in {"", "."}
            else tr("Model library")
        )
        return (label, str(resolved.parent))
    run_folder = relative.parts[0] if relative.parts else resolved.parent.name
    return (recorded or run_folder, str((DEFAULT_OUTPUT_ROOT / run_folder).resolve()))


def _metrics_from_meta(meta: Mapping[str, Any]) -> dict[str, float]:
    raw = meta.get("metrics")
    if not isinstance(raw, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _metrics_from_run_results(path: Path, meta: Mapping[str, Any]) -> dict[str, float]:
    """Recover metrics for a bundle saved before sidecars existed.

    A Benchmark Run writes ``results.csv`` next to its ``models/`` folder, keyed
    by experiment, model, loss, scaler and augmentation, which is exactly the
    information encoded in the bundle file name.
    """

    results = path.parent.parent / "results.csv"
    if not results.exists():
        return {}
    stem_parts = path.stem.split("__")
    if len(stem_parts) < 5:
        return {}
    experiment_key, model_id, loss_id, scaler_id = stem_parts[:4]
    augmentation = stem_parts[4]
    try:
        import pandas as pd

        frame = pd.read_csv(results)
    except Exception:
        return {}
    required = {"experiment_key", "model_id", "loss", "scaler", "augmentation"}
    if not required <= set(frame.columns):
        return {}
    normalized = frame["augmentation"].astype(str).str.replace(":", "_", regex=False).str.replace(
        "+", "_", regex=False
    )
    matched = frame[
        (frame["experiment_key"].astype(str) == experiment_key)
        & (frame["model_id"].astype(str) == model_id)
        & (frame["loss"].astype(str) == loss_id)
        & (frame["scaler"].astype(str) == scaler_id)
        & (normalized == augmentation)
    ]
    if matched.empty:
        return {}
    row = matched.iloc[0]
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in matched.columns:
            try:
                metrics[key] = float(row[key])
            except (TypeError, ValueError):
                continue
    return metrics


# ---------------------------------------------------------------------------
# The user-owned index
# ---------------------------------------------------------------------------

class LibraryIndex:
    """Persisted names, enable flags and notes, keyed by relative model path."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or LIBRARY_INDEX_PATH)
        self._entries: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._entries, self._runs = {}, {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._entries, self._runs = {}, {}
            return
        self._entries = dict(payload.get("models") or {})
        self._runs = dict(payload.get("runs") or {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "runs": self._runs,
            "models": self._entries,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- per-model settings ------------------------------------------------
    def settings(self, entry_id: str) -> dict[str, Any]:
        return dict(self._entries.get(entry_id) or {})

    def set_display_name(self, entry_id: str, name: str) -> None:
        self._entries.setdefault(entry_id, {})["display_name"] = name.strip()

    def set_enabled(self, entry_id: str, enabled: bool) -> None:
        self._entries.setdefault(entry_id, {})["enabled"] = bool(enabled)

    def set_notes(self, entry_id: str, notes: str) -> None:
        self._entries.setdefault(entry_id, {})["notes"] = notes

    def forget(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    # -- per-run settings --------------------------------------------------
    def run_name(self, run_folder: str, default: str) -> str:
        return str((self._runs.get(run_folder) or {}).get("name") or default)

    def set_run_name(self, run_folder: str, name: str) -> None:
        self._runs.setdefault(run_folder, {})["name"] = name.strip()

    def run_notes(self, run_folder: str) -> str:
        return str((self._runs.get(run_folder) or {}).get("notes") or "")

    def set_run_notes(self, run_folder: str, notes: str) -> None:
        self._runs.setdefault(run_folder, {})["notes"] = notes

    def prune(self, known_ids: Iterable[str]) -> None:
        """Drop settings for models that no longer exist on disk."""

        known = set(known_ids)
        for entry_id in list(self._entries):
            if entry_id not in known:
                self._entries.pop(entry_id)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_model_files(roots: Sequence[Path] | None = None) -> list[Path]:
    """Every candidate model file under the library and the run folders."""

    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots or model_search_roots():
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            if path.name.endswith(".meta.json"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def build_entry(path: Path, index: LibraryIndex) -> ModelEntry:
    """Cheap pass: metadata only, no model loading."""

    meta = read_sidecar(path)
    entry_id = _relative_id(path)
    run_label, run_folder = _run_label_for(path, meta)
    run_label = index.run_name(run_folder, run_label)
    metrics = _metrics_from_meta(meta) or _metrics_from_run_results(path, meta)

    model_name = str(meta.get("model_name") or "")
    rank = meta.get("rank")
    target_task_name = str(meta.get("target_task_name") or meta.get("target_task") or "")
    default_parts = [model_name or path.stem]
    if isinstance(rank, int):
        default_parts.append(f"#{rank}")
    if target_task_name:
        # Two Target Tasks in one run can pick the same model family, so the
        # predicted target is what keeps their default names distinguishable.
        default_parts.append(f"→ {target_task_name}")
    default_name = " ".join(default_parts)

    settings = index.settings(entry_id)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return ModelEntry(
        entry_id=entry_id,
        path=path,
        run_label=run_label,
        run_folder=run_folder,
        default_name=default_name,
        display_name=str(settings.get("display_name") or default_name),
        enabled=bool(settings.get("enabled", True)),
        notes=str(settings.get("notes") or ""),
        kind=str(meta.get("kind") or ""),
        model_id=str(meta.get("model_id") or ""),
        model_name=model_name,
        experiment=str(meta.get("experiment") or ""),
        target_task=str(meta.get("target_task_name") or meta.get("target_task") or ""),
        rank=rank if isinstance(rank, int) else None,
        evidence_stage=str(meta.get("evidence_stage") or ""),
        architecture=str(meta.get("architecture") or ""),
        feature_names=tuple(str(name) for name in (meta.get("feature_labels") or meta.get("feature_names") or ())),
        target_names=tuple(str(name) for name in (meta.get("target_labels") or meta.get("target_names") or ())),
        metrics=metrics,
        trained_at=str(meta.get("saved_at") or ""),
        size_bytes=size,
        meta=meta,
    )


def probe_entry(entry: ModelEntry) -> tuple[ModelEntry, Predictor | None]:
    """Deep pass: actually load the model and confirm it can predict."""

    try:
        predictor = load_predictor(entry.path)
    except ModelLoadError as exc:
        return (
            replace(
                entry,
                compatible=False,
                probed=True,
                reason=tr("Incompatible: {reason}", reason=str(exc)),
            ),
            None,
        )
    except Exception as exc:  # a third-party pickle can fail in any way
        return (
            replace(
                entry,
                compatible=False,
                probed=True,
                reason=tr(
                    "Incompatible: loading raised {error_type}: {error}",
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            ),
            None,
        )
    updated = replace(
        entry,
        kind=predictor.kind,
        architecture=predictor.architecture or entry.architecture,
        feature_names=predictor.feature_names,
        target_names=predictor.target_names,
        compatible=True,
        probed=True,
        reason=(
            tr("Runnable ({notes})", notes="; ".join(predictor.notes))
            if predictor.notes
            else tr("Runnable")
        ),
    )
    if not updated.display_name:
        updated.display_name = updated.default_name or entry.path.stem
    return updated, predictor


def scan_library(index: LibraryIndex, *, probe: bool = True) -> Iterator[ModelEntry]:
    """Yield entries as they are discovered so the UI can fill incrementally."""

    for path in discover_model_files():
        entry = build_entry(path, index)
        if probe:
            entry, _predictor = probe_entry(entry)
        yield entry
