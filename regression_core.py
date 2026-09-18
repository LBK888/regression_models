# -*- coding: utf-8 -*-
"""Shared regression core extracted from the v3 monolith.

This module carries only the GUI-free, plotting-free parts that the v4
benchmark engine and the inference layer both need:

* the flexible CSV/XLSX loader and its dataset/column audit model,
* Experiment (input/output combination) generation,
* the legacy PyTorch architectures that trained models are reloaded into,
* the shared device, seeding, early-stopping and batch-size helpers.

Extracted verbatim from ``pytorch_regression_system_v3_0_pyqt.py`` (v3.0) so
that model bundles pickled by either generation keep loading.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import re
import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


FORCE_CPU = os.environ.get("REGRESSION_FORCE_CPU", "").strip() == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
EXECUTION_MODE_LABEL = "GPU mode" if DEVICE.type == "cuda" else "CPU mode"
DEFAULT_SEED = 42
SHEET_ROLE_LABELS = ["Skip", "Input", "Output", "Input or Output"]
COLUMN_ROLE_LABELS = ["Skip", "Combined Input", "Output", "Input or Output"]
AUGMENTATION_METHODS = [
    "gaussian_noise_light",
    "gaussian_noise_moderate",
    "gaussian_noise_heavy",
    "feature_scaling_light",
    "feature_scaling_moderate",
    "mixup",
    "bootstrap",
    "feature_dropout",
    "smote_style",
    "synthetic_interpolation",
]
MODEL_TYPES = ["Deep_FFN", "Wide_Network", "ResNet", "Ensemble_Net", "AutoEncoder_Net"]
SCALER_TYPES = ["StandardScaler", "MinMaxScaler", "RobustScaler"]


# =============================================================================

class DataRole(str, Enum):
    SKIP = "skip"
    INPUT = "input"
    COMBINED_INPUT = "combined_input"
    OUTPUT = "output"
    FLEX = "input_or_output"

    @classmethod
    def from_ui(cls, text: str) -> "DataRole":
        mapping = {
            "Skip": cls.SKIP,
            "Input": cls.INPUT,
            "Combined Input": cls.COMBINED_INPUT,
            "Output": cls.OUTPUT,
            "Input or Output": cls.FLEX,
        }
        return mapping.get(text, cls.SKIP)

    def to_ui(self) -> str:
        return {
            DataRole.SKIP: "Skip",
            DataRole.INPUT: "Input",
            DataRole.COMBINED_INPUT: "Combined Input",
            DataRole.OUTPUT: "Output",
            DataRole.FLEX: "Input or Output",
        }[self]


@dataclass(frozen=True)
class ColumnRef:
    sheet: str
    column: str

    @property
    def canonical_name(self) -> str:
        return f"{self.sheet}::{self.column}"


@dataclass
class ColumnProfile:
    name: str
    non_missing: int
    missing: int
    numeric_values: int
    invalid_numeric: int
    unique_values: int
    numeric_ratio: float
    trainable: bool


@dataclass
class SheetProfile:
    name: str
    original_rows: int
    original_columns: int
    cleaned_rows: int
    cleaned_columns: int
    removed_empty_rows: int
    removed_empty_columns: int
    header_detected: bool
    header_row_index: Optional[int]
    header_confidence: float
    valid_complete_rows: int
    total_missing_cells: int
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)


@dataclass
class LoadedSheet:
    name: str
    data: pd.DataFrame
    profile: SheetProfile


@dataclass
class DatasetBundle:
    source_path: str
    sheets: "OrderedDict[str, LoadedSheet]"
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def trainable_columns(self, sheet_name: str) -> List[str]:
        sheet = self.sheets[sheet_name]
        return [name for name, p in sheet.profile.column_profiles.items() if p.trainable]

    def get_series(self, ref: ColumnRef) -> pd.Series:
        return self.sheets[ref.sheet].data[ref.column]

    def build_numeric_frame(self, refs: Sequence[ColumnRef]) -> pd.DataFrame:
        """Align selected columns by preserved observation id."""
        if not refs:
            return pd.DataFrame()
        series_list: List[pd.Series] = []
        for ref in refs:
            s = pd.to_numeric(self.get_series(ref), errors="coerce")
            s.name = ref.canonical_name
            series_list.append(s)
        frame = pd.concat(series_list, axis=1).sort_index()
        return frame.replace([np.inf, -np.inf], np.nan)

    def audit_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for sheet_name, loaded in self.sheets.items():
            p = loaded.profile
            rows.append(
                {
                    "sheet": sheet_name,
                    "item": "[SHEET]",
                    "trainable": True,
                    "rows": p.cleaned_rows,
                    "missing": p.total_missing_cells,
                    "numeric_ratio": np.nan,
                    "header_detected": p.header_detected,
                    "header_row": p.header_row_index,
                    "removed_empty_rows": p.removed_empty_rows,
                    "removed_empty_columns": p.removed_empty_columns,
                    "notes": f"{p.cleaned_columns} columns",
                }
            )
            for column_name, cp in p.column_profiles.items():
                rows.append(
                    {
                        "sheet": sheet_name,
                        "item": column_name,
                        "trainable": cp.trainable,
                        "rows": p.cleaned_rows,
                        "missing": cp.missing,
                        "numeric_ratio": cp.numeric_ratio,
                        "header_detected": p.header_detected,
                        "header_row": p.header_row_index,
                        "removed_empty_rows": p.removed_empty_rows,
                        "removed_empty_columns": p.removed_empty_columns,
                        "notes": f"invalid numeric={cp.invalid_numeric}, unique={cp.unique_values}",
                    }
                )
        return pd.DataFrame(rows)


class FlexibleDatasetLoader:
    """Load and audit CSV/XLSX files without assuming one fixed layout."""

    MISSING_TOKENS = {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "missing",
        "-",
        "--",
        "—",
        "…",
    }

    def __init__(self, numeric_threshold: float = 0.70, header_scan_rows: int = 12):
        self.numeric_threshold = float(numeric_threshold)
        self.header_scan_rows = int(header_scan_rows)

    def load(self, path: str | os.PathLike[str]) -> DatasetBundle:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in {".xlsx", ".xls", ".csv"}:
            raise ValueError("Only .xlsx, .xls and .csv files are supported")

        if suffix in {".xlsx", ".xls"}:
            raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None, dtype=object)
        else:
            raw_sheets = OrderedDict([(file_path.stem or "CSV", self._read_csv(file_path))])

        sheets: "OrderedDict[str, LoadedSheet]" = OrderedDict()
        for raw_name, raw_df in raw_sheets.items():
            safe_name = self._unique_sheet_name(str(raw_name), sheets)
            sheets[safe_name] = self._clean_sheet(safe_name, raw_df)

        if not sheets:
            raise ValueError("No worksheet or CSV table could be read")
        return DatasetBundle(source_path=str(file_path), sheets=sheets)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        errors: List[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp950", "big5", "latin1"):
            try:
                return pd.read_csv(
                    path,
                    header=None,
                    dtype=object,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                )
            except Exception as exc:
                errors.append(f"{encoding}: {exc}")
        raise ValueError("Unable to read CSV. Tried common encodings. " + " | ".join(errors[-3:]))

    @staticmethod
    def _unique_sheet_name(name: str, existing: Mapping[str, Any]) -> str:
        base = name.strip() or "Sheet"
        candidate = base
        idx = 2
        while candidate in existing:
            candidate = f"{base}_{idx}"
            idx += 1
        return candidate

    @classmethod
    def _normalize_missing(cls, value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in cls.MISSING_TOKENS:
                return np.nan
            return stripped
        return value

    @staticmethod
    def _is_number(value: Any) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return np.isfinite(float(value))
        try:
            float(str(value).strip().replace(",", ""))
            return True
        except Exception:
            return False

    def _detect_header(self, frame: pd.DataFrame) -> Tuple[Optional[int], float]:
        """Score early rows as possible headers. Returns positional row index."""
        if frame.empty:
            return None, 0.0
        max_scan = min(self.header_scan_rows, len(frame))
        best_idx: Optional[int] = None
        best_score = 0.0

        for pos in range(max_scan):
            row = frame.iloc[pos]
            nonempty = [v for v in row.tolist() if not pd.isna(v)]
            if len(nonempty) < max(2, int(frame.shape[1] * 0.20)):
                continue

            string_ratio = np.mean([isinstance(v, str) and not self._is_number(v) for v in nonempty])
            numeric_ratio = np.mean([self._is_number(v) for v in nonempty])
            normalized = [str(v).strip().casefold() for v in nonempty]
            unique_ratio = len(set(normalized)) / max(1, len(normalized))

            after = frame.iloc[pos + 1 : min(len(frame), pos + 6)]
            if after.empty:
                continue
            candidate_columns = [i for i, v in enumerate(row.tolist()) if not pd.isna(v)]
            after_values = after.iloc[:, candidate_columns].to_numpy().ravel().tolist()
            after_nonempty = [v for v in after_values if not pd.isna(v)]
            after_numeric_ratio = (
                np.mean([self._is_number(v) for v in after_nonempty]) if after_nonempty else 0.0
            )

            width_ratio = len(nonempty) / max(1, frame.shape[1])
            score = (
                0.35 * string_ratio
                + 0.35 * after_numeric_ratio
                + 0.15 * unique_ratio
                + 0.15 * width_ratio
                - 0.25 * numeric_ratio
            )
            if score > best_score:
                best_score = float(score)
                best_idx = pos

        if best_idx is not None and best_score >= 0.48:
            return best_idx, min(1.0, best_score)
        return None, max(0.0, best_score)

    @staticmethod
    def _make_unique_headers(values: Sequence[Any]) -> List[str]:
        headers: List[str] = []
        counts: Dict[str, int] = {}
        for idx, value in enumerate(values, start=1):
            if pd.isna(value) or not str(value).strip():
                base = f"Column_{idx}"
            else:
                base = re.sub(r"\s+", " ", str(value).strip())
            counts[base] = counts.get(base, 0) + 1
            headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
        return headers

    def _clean_sheet(self, name: str, raw_df: pd.DataFrame) -> LoadedSheet:
        raw = raw_df.copy()
        original_rows, original_columns = raw.shape
        raw = raw.apply(lambda col: col.map(self._normalize_missing))

        empty_row_mask = raw.isna().all(axis=1)
        empty_col_mask = raw.isna().all(axis=0)
        removed_empty_rows = int(empty_row_mask.sum())
        removed_empty_columns = int(empty_col_mask.sum())
        # Keep empty rows until observation ids have been assigned below. Removing
        # them here would compress one sheet independently and corrupt alignment.
        trimmed = raw.loc[:, ~empty_col_mask].copy()

        if trimmed.empty or trimmed.shape[1] == 0:
            data = pd.DataFrame()
            profile = SheetProfile(
                name=name,
                original_rows=original_rows,
                original_columns=original_columns,
                cleaned_rows=0,
                cleaned_columns=0,
                removed_empty_rows=removed_empty_rows,
                removed_empty_columns=removed_empty_columns,
                header_detected=False,
                header_row_index=None,
                header_confidence=0.0,
                valid_complete_rows=0,
                total_missing_cells=0,
                column_profiles={},
            )
            return LoadedSheet(name=name, data=data, profile=profile)

        header_pos, confidence = self._detect_header(trimmed)
        if header_pos is not None:
            headers = self._make_unique_headers(trimmed.iloc[header_pos].tolist())
            data = trimmed.iloc[header_pos + 1 :].copy()
            header_row_index = int(trimmed.index[header_pos])
            header_detected = True
        else:
            headers = [f"Column_{i + 1}" for i in range(trimmed.shape[1])]
            data = trimmed.copy()
            header_row_index = None
            header_detected = False

        data.columns = headers
        # Observation ids represent data-row positions after the detected header.
        # Empty rows may be removed, but their ids must remain as gaps so another
        # sheet cannot silently shift later observations upward.
        data.index = pd.RangeIndex(len(data), name="observation_id")
        data = data.dropna(how="all")

        column_profiles: Dict[str, ColumnProfile] = OrderedDict()
        converted = pd.DataFrame(index=data.index)
        for column in data.columns:
            original_series = data[column]
            non_missing_mask = original_series.notna()
            non_missing = int(non_missing_mask.sum())
            numeric_series = pd.to_numeric(
                original_series.map(
                    lambda x: str(x).replace(",", "") if isinstance(x, str) else x
                ),
                errors="coerce",
            )
            numeric_values = int(numeric_series.notna().sum())
            invalid_numeric = max(0, non_missing - numeric_values)
            numeric_ratio = numeric_values / non_missing if non_missing else 0.0
            trainable = numeric_values >= 3 and numeric_ratio >= self.numeric_threshold
            converted[column] = numeric_series if trainable else original_series
            column_profiles[column] = ColumnProfile(
                name=column,
                non_missing=non_missing,
                missing=int(len(original_series) - non_missing),
                numeric_values=numeric_values,
                invalid_numeric=invalid_numeric,
                unique_values=int(original_series.nunique(dropna=True)),
                numeric_ratio=float(numeric_ratio),
                trainable=bool(trainable),
            )

        trainable_names = [name for name, p in column_profiles.items() if p.trainable]
        valid_complete_rows = (
            int(converted[trainable_names].notna().all(axis=1).sum()) if trainable_names else 0
        )
        profile = SheetProfile(
            name=name,
            original_rows=original_rows,
            original_columns=original_columns,
            cleaned_rows=len(converted),
            cleaned_columns=converted.shape[1],
            removed_empty_rows=removed_empty_rows,
            removed_empty_columns=removed_empty_columns,
            header_detected=header_detected,
            header_row_index=header_row_index,
            header_confidence=float(confidence),
            valid_complete_rows=valid_complete_rows,
            total_missing_cells=int(converted.isna().sum().sum()),
            column_profiles=column_profiles,
        )
        return LoadedSheet(name=name, data=converted, profile=profile)


# =============================================================================
# Input/output experiment combinations
# =============================================================================

@dataclass(frozen=True)
class SelectionUnit:
    unit_id: str
    label: str
    role: DataRole
    columns: Tuple[ColumnRef, ...]
    scope: str  # "sheet" or "column"


@dataclass
class ExperimentSpec:
    experiment_id: str
    input_units: Tuple[str, ...]
    output_units: Tuple[str, ...]
    input_labels: Tuple[str, ...]
    output_labels: Tuple[str, ...]
    input_columns: Tuple[ColumnRef, ...]
    output_columns: Tuple[ColumnRef, ...]
    valid_rows: int = 0
    dropped_rows: int = 0

    @property
    def display_name(self) -> str:
        return f"{' + '.join(self.input_labels)} -> {' + '.join(self.output_labels)}"


class CombinationLimitError(RuntimeError):
    pass


def _powerset(items: Sequence[SelectionUnit]) -> Iterator[Tuple[SelectionUnit, ...]]:
    for size in range(len(items) + 1):
        yield from itertools.combinations(items, size)


def _flatten_unique_columns(units: Sequence[SelectionUnit]) -> Tuple[ColumnRef, ...]:
    seen: set[ColumnRef] = set()
    output: List[ColumnRef] = []
    for unit in units:
        for ref in unit.columns:
            if ref not in seen:
                seen.add(ref)
                output.append(ref)
    return tuple(output)


def generate_experiment_specs(
    units: Sequence[SelectionUnit],
    bundle: Optional[DatasetBundle] = None,
    max_combinations: int = 0,
    min_valid_rows: int = 8,
) -> List[ExperimentSpec]:
    """
    Generate all unique non-overlapping experiments.

    INPUT and OUTPUT units are eligible members of input/output subsets.
    COMBINED_INPUT units are always attached to another selected input. They may
    stand alone only when the user supplied no other input-capable unit.
    FLEX units have three possible states in each experiment: skipped, input, output.
    At least one input and one output are required.
    """
    fixed_inputs = [u for u in units if u.role == DataRole.INPUT]
    combined_inputs = [u for u in units if u.role == DataRole.COMBINED_INPUT]
    fixed_outputs = [u for u in units if u.role == DataRole.OUTPUT]
    flexible = [u for u in units if u.role == DataRole.FLEX]

    if not units or not (fixed_outputs or flexible):
        return []

    specs: List[ExperimentSpec] = []
    seen: set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    for input_subset in _powerset(fixed_inputs):
        for output_subset in _powerset(fixed_outputs):
            for states in itertools.product((0, 1, 2), repeat=len(flexible)):
                flex_inputs = tuple(u for u, state in zip(flexible, states) if state == 1)
                flex_outputs = tuple(u for u, state in zip(flexible, states) if state == 2)
                other_inputs = tuple(input_subset) + flex_inputs
                if other_inputs:
                    selected_inputs = other_inputs + tuple(combined_inputs)
                elif combined_inputs and not (fixed_inputs or flexible):
                    selected_inputs = tuple(combined_inputs)
                else:
                    selected_inputs = ()
                selected_outputs = tuple(output_subset) + flex_outputs
                if not selected_inputs or not selected_outputs:
                    continue

                input_columns = _flatten_unique_columns(selected_inputs)
                output_columns = _flatten_unique_columns(selected_outputs)
                input_names = {ref.canonical_name for ref in input_columns}
                output_names = {ref.canonical_name for ref in output_columns}
                if input_names & output_names:
                    continue

                key = (tuple(sorted(input_names)), tuple(sorted(output_names)))
                if key in seen:
                    continue
                seen.add(key)

                valid_rows = 0
                dropped_rows = 0
                if bundle is not None:
                    frame = bundle.build_numeric_frame(input_columns + output_columns)
                    valid_rows = int(frame.notna().all(axis=1).sum())
                    dropped_rows = int(len(frame) - valid_rows)
                    if valid_rows < min_valid_rows:
                        continue

                spec = ExperimentSpec(
                    experiment_id=f"EXP_{len(specs) + 1:05d}",
                    input_units=tuple(u.unit_id for u in selected_inputs),
                    output_units=tuple(u.unit_id for u in selected_outputs),
                    input_labels=tuple(u.label for u in selected_inputs),
                    output_labels=tuple(u.label for u in selected_outputs),
                    input_columns=input_columns,
                    output_columns=output_columns,
                    valid_rows=valid_rows,
                    dropped_rows=dropped_rows,
                )
                specs.append(spec)
                if max_combinations > 0 and len(specs) > max_combinations:
                    raise CombinationLimitError(
                        f"Generated more than the configured limit ({max_combinations}). "
                        "Reduce selected units or increase the limit."
                    )
    return specs


def prepare_experiment_arrays(
    bundle: DatasetBundle, spec: ExperimentSpec
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], int]:
    refs = spec.input_columns + spec.output_columns
    frame = bundle.build_numeric_frame(refs)
    valid = frame.notna().all(axis=1)
    clean = frame.loc[valid]
    n_inputs = len(spec.input_columns)
    if clean.empty:
        raise ValueError(f"{spec.experiment_id} has no complete numeric rows")
    X = clean.iloc[:, :n_inputs].to_numpy(dtype=np.float32, copy=True)
    y = clean.iloc[:, n_inputs:].to_numpy(dtype=np.float32, copy=True)
    if X.ndim != 2 or y.ndim != 2 or X.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError(f"Invalid X/y dimensions for {spec.experiment_id}: {X.shape}, {y.shape}")
    return (
        X,
        y,
        [ref.canonical_name for ref in spec.input_columns],
        [ref.canonical_name for ref in spec.output_columns],
        int((~valid).sum()),
    )


# =============================================================================
# Original v2.2 neural-network architectures
# =============================================================================

class DeepFFN(nn.Module):
    """Deep Feedforward Network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (128, 64, 32)):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WideNetwork(nn.Module):
    """Wide network with fewer layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for index, hidden in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.4 if index == 0 else 0.2),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.res_block = ResidualBlock(hidden_dim)
        self.output_layer = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.res_block(self.input_layer(x)))


class EnsembleNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        path1_dim: int = 32,
        path2_dim: int = 64,
        output_hidden: int = 32,
    ):
        super().__init__()
        path1_dim = max(4, path1_dim)
        path2_dim = max(4, path2_dim)
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, path1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(path1_dim, max(2, path1_dim // 2)),
            nn.ReLU(),
        )
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, path2_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        combined = max(2, path1_dim // 2) + path2_dim
        output_hidden = max(4, output_hidden)
        self.output = nn.Sequential(
            nn.Linear(combined, output_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.cat([self.path1(x), self.path2(x)], dim=1))


class AutoEncoderNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bottleneck_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        bottleneck_dim = max(2, bottleneck_dim)
        hidden_dim = max(4, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class ArchitectureSpec:
    name: str
    kwargs: Dict[str, Any]


def _scaled_width(value: int, multiplier: float, minimum: int = 2, maximum: int = 4096) -> int:
    return max(minimum, min(maximum, int(round(value * multiplier))))


def design_architectures(
    input_dim: int,
    output_dim: int,
    data_size: int,
    selected_models: Sequence[str],
    multiplier: float = 1.0,
) -> Dict[str, ArchitectureSpec]:
    selected = set(selected_models)
    architectures: Dict[str, ArchitectureSpec] = OrderedDict()

    if "Deep_FFN" in selected:
        base = [max(4, min(128, data_size * 2)), max(4, min(64, data_size)), max(4, min(32, max(2, data_size // 2)))]
        architectures["Deep_FFN"] = ArchitectureSpec(
            "Deep_FFN", {"hidden_dims": [_scaled_width(v, multiplier, 4) for v in base]}
        )
    if "Wide_Network" in selected:
        base = [max(4, min(256, data_size * 4)), max(4, min(128, data_size * 2))]
        architectures["Wide_Network"] = ArchitectureSpec(
            "Wide_Network", {"hidden_dims": [_scaled_width(v, multiplier, 4) for v in base]}
        )
    if "ResNet" in selected and data_size > 30:
        architectures["ResNet"] = ArchitectureSpec(
            "ResNet", {"hidden_dim": _scaled_width(max(8, min(64, data_size)), multiplier, 4)}
        )
    if "Ensemble_Net" in selected:
        architectures["Ensemble_Net"] = ArchitectureSpec(
            "Ensemble_Net",
            {
                "path1_dim": _scaled_width(max(4, min(32, max(2, data_size // 2))), multiplier, 4),
                "path2_dim": _scaled_width(max(4, min(64, data_size)), multiplier, 4),
                "output_hidden": _scaled_width(32, multiplier, 4),
            },
        )
    if "AutoEncoder_Net" in selected and input_dim > 5:
        architectures["AutoEncoder_Net"] = ArchitectureSpec(
            "AutoEncoder_Net",
            {
                "bottleneck_dim": _scaled_width(max(2, min(16, max(2, data_size // 4))), multiplier, 2),
                "hidden_dim": _scaled_width(64, multiplier, 4),
            },
        )
    return architectures


def build_model(
    architecture: ArchitectureSpec,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    builders: Dict[str, Callable[..., nn.Module]] = {
        "Deep_FFN": DeepFFN,
        "Wide_Network": WideNetwork,
        "ResNet": ResNet,
        "Ensemble_Net": EnsembleNet,
        "AutoEncoder_Net": AutoEncoderNet,
    }
    if architecture.name not in builders:
        raise KeyError(f"Unknown architecture: {architecture.name}")
    return builders[architecture.name](input_dim, output_dim, **architecture.kwargs)


def create_scaler(name: str):
    factories = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }
    if name not in factories:
        raise KeyError(f"Unknown scaler: {name}")
    return factories[name]()


# =============================================================================
# Training, augmentation and evaluation
# =============================================================================

@dataclass
class TrainingConfig:
    use_augmentation: bool = True
    augmentation_strategy: str = "separate"  # separate or combined
    augmentation_methods: List[str] = field(default_factory=lambda: AUGMENTATION_METHODS.copy())
    augmentation_repeats: int = 3
    include_original_in_augmented: bool = False
    epochs: int = 5000
    patience: int = 30
    repetitions: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    test_size: float = 0.20
    validation_split: float = 0.20
    selected_models: List[str] = field(default_factory=lambda: MODEL_TYPES.copy())
    selected_scalers: List[str] = field(default_factory=lambda: SCALER_TYPES.copy())
    model_multiplier: float = 1.0
    random_seed: int = DEFAULT_SEED
    randomize_seed: bool = True
    deterministic_torch: bool = True
    max_combinations: int = 0
    min_valid_rows: int = 8
    save_best_models: bool = True
    save_augmented_snapshot: bool = False
    output_folder: str = ""

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if self.augmentation_repeats < 1:
            raise ValueError("augmentation multiplier must be >= 1")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 because the models use BatchNorm")
        if not 0.05 <= self.test_size <= 0.50:
            raise ValueError("test_size must be between 0.05 and 0.50")
        if not 0.05 <= self.validation_split <= 0.50:
            raise ValueError("validation_split must be between 0.05 and 0.50")
        if not self.selected_models:
            raise ValueError("Select at least one model")
        if not self.selected_scalers:
            raise ValueError("Select at least one scaler")
        invalid_methods = set(self.augmentation_methods) - set(AUGMENTATION_METHODS)
        if invalid_methods:
            raise ValueError(f"Unknown augmentation methods: {sorted(invalid_methods)}")
        if self.use_augmentation and not self.augmentation_methods:
            raise ValueError("Select at least one augmentation method or disable augmentation")
        if self.augmentation_strategy not in {"separate", "combined"}:
            raise ValueError("augmentation_strategy must be separate or combined")

    def augmentation_dataset_count(self) -> int:
        if not self.use_augmentation:
            return 1
        if self.augmentation_strategy == "combined":
            return 2  # original + combined
        return 1 + len(self.augmentation_methods)


@dataclass(frozen=True)
class TrialSeedPlan:
    master_seed: int
    split_seed: int
    validation_seed: int
    augmentation_seed: int


@dataclass(frozen=True)
class SeedLedger:
    """Create stable, recorded seeds without coupling trials to loop order."""

    master_seed: int
    randomized: bool

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "SeedLedger":
        master = secrets.randbits(31) if config.randomize_seed else int(config.random_seed)
        return cls(master_seed=master, randomized=bool(config.randomize_seed))

    def derive(self, namespace: str, *identity: Any) -> int:
        material = "|".join([str(self.master_seed), namespace, *(str(value) for value in identity)])
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF

    def trial(self, experiment_id: str, repetition: int) -> TrialSeedPlan:
        # Splits stay fixed across repetitions so repeats measure training
        # stochasticity rather than silently changing the evaluation cohort.
        return TrialSeedPlan(
            master_seed=self.master_seed,
            split_seed=self.derive("test-split", experiment_id),
            validation_seed=self.derive("validation-split", experiment_id),
            augmentation_seed=self.derive("augmentation", experiment_id, repetition),
        )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def file_sha256(path: str | os.PathLike[str]) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


class TrainingCancelled(RuntimeError):
    pass


class EarlyStopping:
    """Early stopping with a true deep copy of the best state dict."""

    def __init__(self, patience: int = 30, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.stopped_epoch = 0
        self.best_epoch = 0

    def __call__(self, validation_loss: float, model: nn.Module, epoch: int) -> None:
        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            self.counter = 0
            self.best_epoch = epoch + 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch + 1


def _safe_batch_size(n_samples: int, requested: int) -> int:
    if n_samples < 2:
        raise ValueError("At least two training samples are required")
    batch = max(2, min(requested, n_samples))
    while batch > 2 and n_samples % batch == 1:
        batch -= 1
    if batch == 2 and n_samples > 2 and n_samples % batch == 1:
        batch = n_samples
    return batch

