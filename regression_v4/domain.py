"""Domain types shared by V4's UI, evaluation and reporting layers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


def _canonical_names(values: Iterable[str]) -> tuple[str, ...]:
    """Return a deterministic set-like representation of column names."""

    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _ordered_unique_names(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _short_hash(namespace: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"namespace": namespace, **payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class EvaluationStrategy(str, Enum):
    NESTED_CV = "nested_cv"
    HOLDOUT = "holdout"
    EXTERNAL_VALIDATION = "external_validation"
    TRAIN_FINAL_ONLY = "train_final_only"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PROMISING_UNCERTAIN = "promising_uncertain"
    INCONCLUSIVE = "inconclusive"
    WORSE_THAN_BASELINE = "worse_than_baseline"
    NO_GENERALIZATION_EVIDENCE = "no_generalization_evidence"


class FeatureStructure(str, Enum):
    TABULAR = "tabular"
    ORDERED_1D = "ordered_1d"
    GROUPED = "grouped"
    GROUPED_ORDERED_1D = "grouped_ordered_1d"


def infer_feature_metadata(
    input_columns: Sequence[str],
) -> tuple[FeatureStructure, tuple[str, ...]]:
    groups: list[str] = []
    coordinates_by_group: dict[str, list[float | None]] = {}
    for column in input_columns:
        group = column.rsplit("::", 1)[0] if "::" in column else "__ungrouped__"
        if group not in coordinates_by_group:
            groups.append(group)
            coordinates_by_group[group] = []
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:nm)?$", column, re.IGNORECASE)
        coordinates_by_group[group].append(float(match.group(1)) if match else None)

    def ordered(values: Sequence[float | None]) -> bool:
        return (
            len(values) >= 4
            and all(value is not None for value in values)
            and len(set(values)) == len(values)
            and all(float(right) > float(left) for left, right in zip(values, values[1:]))
        )

    all_ordered = bool(groups) and all(ordered(coordinates_by_group[group]) for group in groups)
    if len(groups) > 1 and all_ordered:
        structure = FeatureStructure.GROUPED_ORDERED_1D
    elif len(groups) > 1:
        structure = FeatureStructure.GROUPED
    elif all_ordered:
        structure = FeatureStructure.ORDERED_1D
    else:
        structure = FeatureStructure.TABULAR
    return structure, tuple(groups)


@dataclass(frozen=True)
class TargetTask:
    """A comparison boundary: models are ranked only inside one output set."""

    output_columns: tuple[str, ...]
    key: str = field(init=False)

    def __post_init__(self) -> None:
        columns = _canonical_names(self.output_columns)
        if not columns:
            raise ValueError("A target task requires at least one output column")
        object.__setattr__(self, "output_columns", columns)
        object.__setattr__(self, "key", f"TASK_{_short_hash('target-task', {'outputs': columns})}")

    @property
    def display_name(self) -> str:
        return " + ".join(self.output_columns)


@dataclass(frozen=True)
class ExperimentRecord:
    """Stable experiment identity separated from its reorderable display label."""

    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    input_labels: tuple[str, ...] = ()
    output_labels: tuple[str, ...] = ()
    valid_rows: int = 0
    dropped_rows: int = 0
    selected: bool = True
    feature_structure: FeatureStructure | str | None = None
    feature_groups: tuple[str, ...] = ()
    structure_confirmed: bool = False
    stable_key: str = field(init=False)
    target_task: TargetTask = field(init=False)

    def __post_init__(self) -> None:
        inputs = _ordered_unique_names(self.input_columns)
        outputs = _ordered_unique_names(self.output_columns)
        if not inputs:
            raise ValueError("An experiment requires at least one input column")
        if not outputs:
            raise ValueError("An experiment requires at least one output column")
        overlap = set(inputs).intersection(outputs)
        if overlap:
            raise ValueError(f"Input/output columns overlap: {sorted(overlap)!r}")
        object.__setattr__(self, "input_columns", inputs)
        object.__setattr__(self, "output_columns", outputs)
        suggested_structure, suggested_groups = infer_feature_metadata(inputs)
        structure = self.feature_structure or suggested_structure
        if not isinstance(structure, FeatureStructure):
            structure = FeatureStructure(str(structure))
        object.__setattr__(self, "feature_structure", structure)
        object.__setattr__(self, "feature_groups", self.feature_groups or suggested_groups)
        object.__setattr__(self, "target_task", TargetTask(outputs))
        object.__setattr__(
            self,
            "stable_key",
            f"EXPK_{_short_hash('experiment', {'inputs': _canonical_names(inputs), 'outputs': _canonical_names(outputs)})}",
        )

    @property
    def display_name(self) -> str:
        inputs = self.input_labels or self.input_columns
        outputs = self.output_labels or self.output_columns
        return f"{' + '.join(inputs)} -> {' + '.join(outputs)}"

    def with_selected(self, selected: bool) -> "ExperimentRecord":
        return replace(self, selected=bool(selected))

    def with_feature_structure(
        self, structure: FeatureStructure | str, *, confirmed: bool
    ) -> "ExperimentRecord":
        return replace(
            self,
            feature_structure=FeatureStructure(structure),
            structure_confirmed=bool(confirmed),
        )

    @classmethod
    def from_legacy(cls, spec: Any, *, selected: bool = True) -> "ExperimentRecord":
        """Convert a V3 ExperimentSpec without importing the monolithic GUI."""

        def name(ref: Any) -> str:
            canonical = getattr(ref, "canonical_name", None)
            return str(canonical if canonical is not None else ref)

        return cls(
            input_columns=tuple(name(ref) for ref in spec.input_columns),
            output_columns=tuple(name(ref) for ref in spec.output_columns),
            input_labels=tuple(spec.input_labels),
            output_labels=tuple(spec.output_labels),
            valid_rows=int(getattr(spec, "valid_rows", 0)),
            dropped_rows=int(getattr(spec, "dropped_rows", 0)),
            selected=selected,
        )


class ExperimentCatalog:
    """Owns experiment selection and preserves it across regeneration."""

    def __init__(self, records: Sequence[ExperimentRecord] = ()) -> None:
        self._records: list[ExperimentRecord] = []
        self.replace(records)

    @property
    def records(self) -> tuple[ExperimentRecord, ...]:
        return tuple(self._records)

    @property
    def selected_records(self) -> tuple[ExperimentRecord, ...]:
        return tuple(record for record in self._records if record.selected)

    def replace(self, records: Sequence[ExperimentRecord]) -> None:
        previous = {record.stable_key: record for record in self._records}
        seen: set[str] = set()
        replacement: list[ExperimentRecord] = []
        for record in records:
            if record.stable_key in seen:
                continue
            seen.add(record.stable_key)
            old = previous.get(record.stable_key)
            if old is not None:
                record = replace(
                    record,
                    selected=old.selected,
                    feature_structure=old.feature_structure,
                    feature_groups=old.feature_groups,
                    structure_confirmed=old.structure_confirmed,
                )
            replacement.append(record)
        self._records = replacement

    def set_selected(self, stable_key: str, selected: bool) -> None:
        for index, record in enumerate(self._records):
            if record.stable_key == stable_key:
                self._records[index] = record.with_selected(selected)
                return
        raise KeyError(stable_key)

    def set_feature_structure(
        self,
        stable_key: str,
        structure: FeatureStructure | str,
        *,
        confirmed: bool,
    ) -> None:
        for index, record in enumerate(self._records):
            if record.stable_key == stable_key:
                self._records[index] = record.with_feature_structure(structure, confirmed=confirmed)
                return
        raise KeyError(stable_key)

    def select_all(self) -> None:
        self._records = [record.with_selected(True) for record in self._records]

    def select_none(self) -> None:
        self._records = [record.with_selected(False) for record in self._records]

    def invert_selection(self) -> None:
        self._records = [record.with_selected(not record.selected) for record in self._records]


@dataclass(frozen=True)
class EvaluationConfig:
    strategy: EvaluationStrategy = EvaluationStrategy.NESTED_CV
    test_size: float = 0.20
    screening_folds: int = 3
    confirmation_folds: int = 5
    confirmation_repeats: int = 3
    confirmation_inner_folds: int = 3
    confirmation_tuning_budget: int = 8
    finalists_per_task: int = 5
    run_confirmation: bool = False
    confirmation_augmentation_ablation: bool = True
    random_seed: int = 42
    refit_on_all_data: bool = True
    epochs: int = 500
    patience: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    model_multiplier: float = 1.0
    repetitions: int = 1
    validation_ratio: float = 0.20
    selected_scalers: tuple[str, ...] = ("standard",)
    selected_losses: tuple[str, ...] = ("mse", "huber")
    augmentation_policy: str = "original_only"
    selected_augmentations: tuple[str, ...] = ()
    augmentation_repeats: int = 1
    augmentation_ratio: float = 0.50
    augmentation_include_original: bool = True
    c_mixup_alpha: float = 2.0
    noise_fraction: float = 0.02
    feature_mask_probability: float = 0.05
    foma_alpha: float = 2.0
    foma_k: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.test_size <= 0.50:
            raise ValueError("test_size must be between 0 and 0.50")
        if self.screening_folds < 2:
            raise ValueError("screening_folds must be at least 2")
        if self.confirmation_folds < 2:
            raise ValueError("confirmation_folds must be at least 2")
        if self.confirmation_repeats < 1:
            raise ValueError("confirmation_repeats must be at least 1")
        if self.confirmation_inner_folds < 2:
            raise ValueError("confirmation_inner_folds must be at least 2")
        if self.confirmation_tuning_budget < 1 or self.finalists_per_task < 1:
            raise ValueError("Confirmation tuning budget and finalist cap must be positive")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 2:
            raise ValueError("epochs/patience must be positive and batch_size must be at least 2")
        if self.learning_rate <= 0 or self.model_multiplier <= 0:
            raise ValueError("learning_rate and model_multiplier must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0.05 <= self.validation_ratio <= 0.50:
            raise ValueError("validation_ratio must be between 0.05 and 0.50")
        allowed_scalers = {"standard", "robust", "minmax"}
        if not self.selected_scalers or not set(self.selected_scalers) <= allowed_scalers:
            raise ValueError("Select at least one supported scaler")
        allowed_losses = {"mse", "huber", "mae", "log_cosh"}
        if not self.selected_losses or not set(self.selected_losses) <= allowed_losses:
            raise ValueError("Select at least one supported loss")
        if self.augmentation_policy not in {"original_only", "compare_separately", "combined"}:
            raise ValueError("Unsupported augmentation_policy")
        allowed_augmentations = {
            "c_mixup",
            "calibrated_noise",
            "spectral_perturbation",
            "feature_masking",
            "foma",
        }
        if not set(self.selected_augmentations) <= allowed_augmentations:
            raise ValueError("Unsupported augmentation method")
        if self.augmentation_policy != "original_only" and not self.selected_augmentations:
            raise ValueError("Select at least one augmentation or use Original only")
        if self.augmentation_repeats < 1 or not 0 < self.augmentation_ratio <= 5:
            raise ValueError("Invalid augmentation repeats/ratio")
        if self.c_mixup_alpha <= 0 or self.noise_fraction < 0:
            raise ValueError("Invalid augmentation strength")
        if not 0 <= self.feature_mask_probability < 1:
            raise ValueError("feature_mask_probability must be in [0, 1)")
        if self.foma_alpha <= 0 or self.foma_k < 1:
            raise ValueError("FOMA alpha must be positive and FOMA k must be at least 1")

    @property
    def has_test_evidence(self) -> bool:
        if self.strategy == EvaluationStrategy.HOLDOUT:
            return self.test_size > 0.0
        return self.strategy in {
            EvaluationStrategy.NESTED_CV,
            EvaluationStrategy.EXTERNAL_VALIDATION,
        }

    @property
    def evidence_note(self) -> str:
        if self.strategy == EvaluationStrategy.TRAIN_FINAL_ONLY:
            return "Final fit only; no generalization evidence was measured."
        if self.strategy == EvaluationStrategy.HOLDOUT and self.test_size == 0:
            return "Holdout is 0%; all rows may be used for fitting, but no test evidence exists."
        return "Generalization evidence is produced by the configured evaluation strategy."


@dataclass(frozen=True)
class ProgressState:
    current: int
    total: int
    experiment_label: str
    stage_label: str = ""

    @property
    def title(self) -> str:
        stage = f" | {self.stage_label}" if self.stage_label else ""
        return f"{self.current}/{self.total}: {self.experiment_label}{stage}"
