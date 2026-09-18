"""PyQt6 user interface for the first executable V4 benchmark slice."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import replace

import numpy as np

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QSizePolicy,
    QTabWidget,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from i18n import tr
from project_paths import DEFAULT_OUTPUT_ROOT
from regression_core import (
    ColumnRef,
    DataRole,
    DatasetBundle,
    EXECUTION_MODE_LABEL,
    FlexibleDatasetLoader,
    SelectionUnit,
    generate_experiment_specs,
)

from .confirmation import (
    ConfirmationConfig,
    ConfirmationResult,
    confirm_finalist,
    promote_finalists,
    representative_parameters,
)
from .domain import (
    EvaluationConfig,
    EvaluationStrategy,
    ExperimentCatalog,
    ExperimentRecord,
    FeatureStructure,
    ProgressState,
)
from .evaluation import (
    EvaluationResult,
    MODEL_KEEP_LIMIT,
    augmentation_incompatibility_reason,
    bundle_metadata,
    evaluate_external_model,
    evaluate_holdout,
    evaluate_screening,
    evaluation_run_count,
    fit_final_models,
    format_trained_epochs,
    save_fitted_models,
    select_retained_results,
)
from .models import (
    ModelAvailability,
    model_incompatibility_reason,
    model_registry,
)
from .reporting import MasterReportBuilder, ReportArtifacts, build_no_evidence_report
from .splitting import DatasetIdentity


def role_options() -> dict[str, str]:
    """Sheet and column role labels, resolved per build so i18n applies."""

    return {
        "skip": tr("Skip"),
        "input": tr("Input"),
        "combined_input": tr("Combined input"),
        "output": tr("Output"),
        "input_or_output": tr("Input or output"),
        "observation_id": tr("Observation ID (not a feature)"),
        "group_id": tr("Group ID (not a feature)"),
    }

HELP_GLYPH = "⍰"


def model_help() -> dict[str, str]:
    """One sentence per model family: what ticking it buys, and what it costs."""

    return {
        "mean": tr(
            "Mean baseline: predicts the training targets' mean and nothing else. Tick "
            "it to see whether a complex model really beats the simplest possible "
            "prediction."
        ),
        "ridge": tr(
            "Ridge linear regression: L2 regularization shrinks unstable coefficients, "
            "which makes the linear fit more robust to collinear features."
        ),
        "pls": tr(
            "PLS: compresses highly correlated features into a few latent components "
            "related to the target. Suited to small data with many collinear features."
        ),
        "svr_rbf": tr(
            "RBF-SVR: learns a nonlinear relationship through a kernel while bounding "
            "the error band. Adds one nonlinear small-sample comparison."
        ),
        "random_forest": tr(
            "Random Forest: averages many randomized decision trees. Captures "
            "nonlinearity and interactions, but takes longer to train."
        ),
        "extra_trees": tr(
            "Extra Trees: builds a tree ensemble with more random splits. Usually "
            "faster, and reduces the variance of any single tree."
        ),
        "deep_mlp": tr(
            "Deep MLP: a multi-layer fully connected network. Learns general "
            "nonlinearity and honours the neural loss you selected."
        ),
        "wide_mlp": tr(
            "Wide MLP: a wider fully connected network. More representational capacity "
            "per layer, at the cost of more parameters and compute."
        ),
        "residual_mlp": tr(
            "Residual MLP: residual connections let a deeper network train, improving "
            "gradient flow and adding nonlinear capacity."
        ),
        "multi_branch_mlp": tr(
            "Multi-branch MLP: several parallel paths extract different representations "
            "before merging. More capacity, and more compute."
        ),
        "bottleneck_mlp": tr(
            "Bottleneck MLP: compresses then rebuilds a higher-level representation, "
            "pushing the model towards a more compact feature combination."
        ),
        "cnn1d": tr(
            "CNN1D: slides a convolution kernel along the ordered 1D features you "
            "confirmed. Only appropriate when neighbouring columns really are locally "
            "related."
        ),
        "resnet1d": tr(
            "ResNet1D: residual convolution blocks over ordered 1D features. Suited to "
            "sequence-like data; spectral column naming is not required."
        ),
        "grouped_fusion": tr(
            "Grouped Fusion: a separate encoder per Feature Group, then a learned "
            "fusion. Runs only when there are at least two meaningful groups."
        ),
        "mlp_embeddings": tr(
            "Numerical-Embedding MLP: turns each numeric feature into a learnable "
            "representation before the MLP. Captures general tabular nonlinearity."
        ),
        "ft_transformer": tr(
            "FT-Transformer: treats each feature as a token and models feature "
            "interaction with attention. Good for general tabular data, but slower."
        ),
        "modern_nca": tr(
            "ModernNCA: learns an embedding where similar targets sit close together, "
            "then predicts from neighbours. Useful for exploring local sample structure."
        ),
        "tabm": tr(
            "TabM: a parameter-shared multi-member tabular network ensemble. Runs only "
            "when the optional package is installed."
        ),
        "realmlp": tr(
            "RealMLP: pytabkit's tuned tabular MLP configuration. Joins the comparison "
            "when the optional package is installed."
        ),
        "catboost": tr(
            "CatBoost: gradient-boosted decision trees, strong on general tabular "
            "nonlinearity. Needs the optional package installed."
        ),
        "xgboost": tr(
            "XGBoost: regularized gradient-boosted trees. Adds the common tabular "
            "boosting comparison, and more compute."
        ),
        "lightgbm": tr(
            "LightGBM: leaf-wise gradient-boosted trees, usually fast. Needs the "
            "optional package installed."
        ),
    }


def loss_help() -> dict[str, str]:
    """What each neural loss emphasises, and what ticking it adds to the run."""

    return {
        "mse": tr(
            "MSE: squares the error, so large outlying deviations dominate. Ticking it "
            "trains one more MSE configuration for every neural model."
        ),
        "huber": tr(
            "Huber / SmoothL1: squared for small errors, roughly linear for large ones, "
            "which limits how much outliers steer neural training."
        ),
        "mae": tr(
            "MAE / L1: every absolute error counts in proportion. Less sensitive to "
            "outliers, but the gradient is less smooth."
        ),
        "log_cosh": tr(
            "Log-cosh: close to MSE for small errors and to MAE for large ones — a "
            "smooth, outlier-tolerant compromise."
        ),
    }


def augmentation_help() -> dict[str, str]:
    """What each training-only augmentation assumes about plausible new data."""

    return {
        "c_mixup": tr(
            "C-Mixup: picks two training rows with similar target values and "
            "interpolates both X and y, producing target-aware synthetic rows."
        ),
        "calibrated_noise": tr(
            "Calibrated noise: adds small random noise to X, scaled by how spread the "
            "training features are, and keeps y at its source value. Models measurement "
            "variation."
        ),
        "spectral_perturbation": tr(
            "Spectral perturbation: adds smooth baseline, multiplicative and noise "
            "variation along a valid wavelength axis. Non-spectral Experiments are "
            "skipped automatically."
        ),
        "feature_masking": tr(
            "Feature masking: randomly masks some scaled features during training, so "
            "the model cannot lean too hard on one column."
        ),
        "foma": tr(
            "FOMA: takes an SVD of the fold-training joint X/y and shrinks the "
            "non-leading manifold components, producing synthetic rows near the "
            "manifold."
        ),
    }


class DatasetDropFrame(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        label = QLabel(tr("Drag a CSV / Excel file here, or use the button below"))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setMinimumHeight(70)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in {".csv", ".xlsx", ".xls"}:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


class CollapsibleSection(QWidget):
    """A compact settings section whose values remain intact while collapsed."""

    def __init__(self, title: str, *, expanded: bool = False) -> None:
        super().__init__()
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setStyleSheet("text-align: left; font-weight: 600; padding: 6px;")
        self.content_widget = QWidget()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)
        self.toggle_button.toggled.connect(self._set_expanded)
        self._set_expanded(expanded)

    def setContentLayout(self, content_layout: Any) -> None:
        self.content_widget.setLayout(content_layout)

    def _set_expanded(self, expanded: bool) -> None:
        self.content_widget.setMaximumHeight(16_777_215 if expanded else 0)
        self.content_widget.setVisible(expanded)
        self.toggle_button.setText(f"{'▼' if expanded else '▶'} {self._title}")
        self.setMaximumHeight(
            16_777_215 if expanded else self.toggle_button.sizeHint().height()
        )
        if self.layout() is not None:
            self.layout().invalidate()
            self.layout().activate()
        self.updateGeometry()
        if self.parentWidget() is not None:
            self.parentWidget().updateGeometry()

    @property
    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()


class ExperimentTableModel(QAbstractTableModel):
    COLUMN_COUNT = 11

    @staticmethod
    def headers() -> tuple[str, ...]:
        """Column labels, resolved per call so a language switch re-renders them."""

        return (
            tr("Use"),
            "ID",
            "Inputs",
            "Outputs",
            tr("Valid rows"),
            tr("Input columns"),
            tr("Output columns"),
            "Stable Key",
            "Target Task",
            "Feature Structure",
            tr("Structure confirmed"),
        )

    def __init__(self, catalog: ExperimentCatalog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.catalog = catalog

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.catalog.records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self.COLUMN_COUNT

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers()[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        record = self.catalog.records[index.row()]
        column = index.column()
        if column == 0 and role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if record.selected else Qt.CheckState.Unchecked
        if column == 10 and role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if record.structure_confirmed else Qt.CheckState.Unchecked
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            values = (
                "",
                f"EXP_{index.row() + 1:05d}",
                " + ".join(record.input_labels or record.input_columns),
                " + ".join(record.output_labels or record.output_columns),
                record.valid_rows,
                len(record.input_columns),
                len(record.output_columns),
                record.stable_key,
                record.target_task.key,
                record.feature_structure.value,
                "",
            )
            return values[column]
        if role == Qt.ItemDataRole.ToolTipRole:
            if column == 2:
                return "\n".join(record.input_columns)
            if column == 3:
                return "\n".join(record.output_columns)
            if column == 0:
                return tr(
                    "Untick a combination to leave it out of training. Regenerating the "
                    "combinations keeps your choice, matched by Stable Key."
                )
            if column == 9:
                return (
                    f"Feature Groups: {', '.join(record.feature_groups)}\n"
                    + tr(
                        "You can change the automatic suggestion. CNN1D, ResNet1D and "
                        "Grouped Fusion only run on combinations that are both confirmed "
                        "and compatible."
                    )
                )
            if column == 10:
                return tr(
                    "Tick this only after confirming the Feature Structure matches what "
                    "the data actually means."
                )
        if role == Qt.ItemDataRole.UserRole:
            return record.stable_key
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        elif index.column() == 9:
            flags |= Qt.ItemFlag.ItemIsEditable
        elif index.column() == 10:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            record = self.catalog.records[index.row()]
            checked = value == Qt.CheckState.Checked or value == Qt.CheckState.Checked.value
            self.catalog.set_selected(record.stable_key, checked)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        if index.isValid() and index.column() == 9 and role == Qt.ItemDataRole.EditRole:
            record = self.catalog.records[index.row()]
            self.catalog.set_feature_structure(record.stable_key, str(value), confirmed=True)
            self.dataChanged.emit(
                index,
                self.index(index.row(), 10),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.CheckStateRole],
            )
            return True
        if index.isValid() and index.column() == 10 and role == Qt.ItemDataRole.CheckStateRole:
            record = self.catalog.records[index.row()]
            checked = value == Qt.CheckState.Checked or value == Qt.CheckState.Checked.value
            self.catalog.set_feature_structure(
                record.stable_key,
                record.feature_structure,
                confirmed=checked,
            )
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        return False

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def set_all(self, mode: str) -> None:
        if mode == "all":
            self.catalog.select_all()
        elif mode == "none":
            self.catalog.select_none()
        elif mode == "invert":
            self.catalog.invert_selection()
        else:
            raise ValueError(mode)
        self.refresh()


class FeatureStructureDelegate(QStyledItemDelegate):
    def createEditor(self, parent: QWidget, option: Any, index: QModelIndex) -> QWidget:
        editor = QComboBox(parent)
        for structure in FeatureStructure:
            editor.addItem(structure.value, structure.value)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        assert isinstance(editor, QComboBox)
        position = editor.findData(index.data(Qt.ItemDataRole.EditRole))
        editor.setCurrentIndex(max(0, position))

    def setModelData(self, editor: QWidget, model: Any, index: QModelIndex) -> None:
        assert isinstance(editor, QComboBox)
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


def prepare_experiment_data(
    bundle: DatasetBundle,
    spec: Any,
    observation_ref: ColumnRef | None,
    group_ref: ColumnRef | None,
) -> tuple[np.ndarray, np.ndarray, DatasetIdentity, int]:
    refs = spec.input_columns + spec.output_columns
    frame = bundle.build_numeric_frame(refs)
    valid = frame.notna().all(axis=1)
    clean = frame.loc[valid]
    n_inputs = len(spec.input_columns)
    X = clean.iloc[:, :n_inputs].to_numpy(dtype=float, copy=True)
    y = clean.iloc[:, n_inputs:].to_numpy(dtype=float, copy=True)
    if observation_ref is None:
        observation_ids = np.asarray([f"row::{index}" for index in clean.index], dtype=object)
    else:
        observation_series = bundle.get_series(observation_ref).reindex(clean.index)
        if observation_series.isna().any():
            raise ValueError(f"Observation ID {observation_ref.canonical_name} contains missing values")
        observation_ids = observation_series.astype(str).to_numpy(dtype=object)
    group_ids = None
    if group_ref is not None:
        group_series = bundle.get_series(group_ref).reindex(clean.index)
        if group_series.isna().any():
            raise ValueError(f"Group ID {group_ref.canonical_name} contains missing values")
        group_ids = group_series.astype(str).to_numpy(dtype=object)
    return X, y, DatasetIdentity(observation_ids, group_ids), int((~valid).sum())


def prepare_external_experiment_data(
    bundle: DatasetBundle,
    spec: Any,
    column_mapping: Mapping[str, ColumnRef],
    observation_ref: ColumnRef | None = None,
) -> tuple[np.ndarray, np.ndarray, DatasetIdentity, int]:
    """Build an external X/y matrix from an explicit development-to-external mapping."""

    development_refs = tuple(spec.input_columns) + tuple(spec.output_columns)
    missing = [ref.canonical_name for ref in development_refs if ref.canonical_name not in column_mapping]
    if missing:
        raise ValueError("External mapping is incomplete: " + ", ".join(missing))
    mapped_refs = tuple(column_mapping[ref.canonical_name] for ref in development_refs)
    frame = bundle.build_numeric_frame(mapped_refs)
    valid = frame.notna().all(axis=1)
    clean = frame.loc[valid]
    n_inputs = len(spec.input_columns)
    X = clean.iloc[:, :n_inputs].to_numpy(dtype=float, copy=True)
    y = clean.iloc[:, n_inputs:].to_numpy(dtype=float, copy=True)
    if observation_ref is None:
        observation_ids = np.asarray([f"external-row::{index}" for index in clean.index], dtype=object)
    else:
        observation_series = bundle.get_series(observation_ref).reindex(clean.index)
        if observation_series.isna().any():
            raise ValueError(
                f"External Observation ID {observation_ref.canonical_name} contains missing values"
            )
        observation_ids = observation_series.astype(str).to_numpy(dtype=object)
    return X, y, DatasetIdentity(observation_ids), int((~valid).sum())


def augmentation_methods_from_id(augmentation_id: str) -> tuple[str, ...]:
    if augmentation_id == "original":
        return ()
    if augmentation_id.startswith("combined:"):
        return tuple(item for item in augmentation_id.removeprefix("combined:").split("+") if item)
    return (augmentation_id,)


class BenchmarkWorker(QThread):
    log_message = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        bundle: DatasetBundle,
        jobs: Sequence[tuple[Any, ExperimentRecord]],
        observation_ref: ColumnRef | None,
        group_ref: ColumnRef | None,
        model_ids: Sequence[str],
        config: EvaluationConfig,
        output_folder: str,
        external_bundle: DatasetBundle | None = None,
        external_mapping: Mapping[str, ColumnRef] | None = None,
        external_observation_ref: ColumnRef | None = None,
        run_label: str = "",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.jobs = tuple(jobs)
        self.observation_ref = observation_ref
        self.group_ref = group_ref
        self.model_ids = tuple(model_ids)
        self.config = config
        self.output_folder = output_folder
        self.external_bundle = external_bundle
        self.external_mapping = dict(external_mapping or {})
        self.external_observation_ref = external_observation_ref
        self.run_label = run_label or Path(output_folder).name
        self.failure_rows: list[dict[str, object]] = []

    def _model_reason(self, model_id: str, record: ExperimentRecord) -> str | None:
        return model_incompatibility_reason(
            model_id,
            record.input_columns,
            record.feature_structure.value,
            record.structure_confirmed,
        )

    def _record_skip(
        self,
        record: ExperimentRecord,
        model_id: str,
        reason: str,
        *,
        stage: str = "compatibility",
    ) -> None:
        row = {
            "stage": stage,
            "experiment_key": record.stable_key,
            "target_task": record.target_task.key,
            "model_id": model_id,
            "reason": reason,
        }
        if row not in self.failure_rows:
            self.failure_rows.append(row)

    def _record_augmentation_skip(
        self,
        record: ExperimentRecord,
        method: str,
        reason: str,
    ) -> None:
        row = {
            "stage": "augmentation_compatibility",
            "experiment_key": record.stable_key,
            "target_task": record.target_task.key,
            "model_id": "",
            "augmentation": method,
            "reason": reason,
        }
        if row not in self.failure_rows:
            self.failure_rows.append(row)

    def _experiment_audit_rows(self) -> list[dict[str, object]]:
        return [
            {
                "experiment_id": spec.experiment_id,
                "experiment_key": record.stable_key,
                "selected": record.selected,
                "target_task": record.target_task.key,
                "inputs": " | ".join(record.input_columns),
                "outputs": " | ".join(record.output_columns),
                "valid_rows": record.valid_rows,
                "feature_structure": record.feature_structure.value,
                "structure_confirmed": record.structure_confirmed,
                "feature_groups": " | ".join(record.feature_groups),
            }
            for spec, record in self.jobs
        ]

    def run(self) -> None:
        try:
            if self.config.strategy == EvaluationStrategy.EXTERNAL_VALIDATION and self.external_bundle is None:
                raise ValueError("External Validation requires a separately loaded external dataset")
            no_evidence_run = self.config.strategy == EvaluationStrategy.TRAIN_FINAL_ONLY or (
                self.config.strategy == EvaluationStrategy.HOLDOUT and self.config.test_size == 0
            )
            if no_evidence_run:
                compatible_by_job = [
                    tuple(
                        model_id for model_id in self.model_ids
                        if self._model_reason(model_id, record) is None
                    )
                    for _spec, record in self.jobs
                ]
                steps_by_job = [
                    evaluation_run_count(model_ids, self.config, 1, record)
                    for (_spec, record), model_ids in zip(self.jobs, compatible_by_job)
                ]
                total = sum(steps_by_job)
                fitted_paths: list[Path] = []
                offset = 0
                for (spec, record), compatible_models, job_steps in zip(
                    self.jobs, compatible_by_job, steps_by_job
                ):
                    for model_id in self.model_ids:
                        reason = self._model_reason(model_id, record)
                        if reason:
                            self._record_skip(record, model_id, reason)
                            self.log_message.emit(f"[{spec.experiment_id}] Skip {model_id}: {reason}")
                    if not compatible_models:
                        continue
                    for method in self.config.selected_augmentations:
                        augmentation_reason = augmentation_incompatibility_reason(method, record)
                        if augmentation_reason:
                            self._record_augmentation_skip(record, method, augmentation_reason)
                            self.log_message.emit(
                                f"[{spec.experiment_id}] Skip augmentation {method}: "
                                f"{augmentation_reason}; other augmentation variants remain active"
                            )
                    X, y, _identity, dropped = prepare_experiment_data(
                        self.bundle, spec, self.observation_ref, self.group_ref
                    )
                    self.log_message.emit(
                        f"[{spec.experiment_id}] final fit, rows={len(X)}, dropped={dropped}; no evaluation metrics"
                    )
                    def final_progress(state: ProgressState, experiment_id: str = spec.experiment_id) -> None:
                        if self.isInterruptionRequested():
                            raise InterruptedError
                        self.progress_changed.emit(
                            state.current,
                            state.total,
                            f"{experiment_id}: {record.display_name} | {state.stage_label}",
                        )

                    bundles = fit_final_models(
                        record,
                        X,
                        y,
                        compatible_models,
                        random_seed=self.config.random_seed,
                        evidence_note=self.config.evidence_note,
                        progress_callback=final_progress,
                        progress_offset=offset,
                        progress_total=total,
                        config=self.config,
                    )
                    for bundle in bundles:
                        epoch_text = format_trained_epochs(
                            (bundle.trained_epochs,), self.config.epochs
                        )
                        if epoch_text:
                            self.log_message.emit(
                                f"  {bundle.model_name} final fit: {epoch_text}"
                            )
                    fitted_paths.extend(
                        save_fitted_models(
                            bundles,
                            self.output_folder,
                            [
                                bundle_metadata(
                                    bundle,
                                    evidence_stage="final_fit_only",
                                    extra={"run_label": self.run_label},
                                )
                                for bundle in bundles
                            ],
                        )
                    )
                    offset += job_steps
                if not fitted_paths:
                    raise ValueError("No selected model is compatible with the selected experiments")
                self.completed.emit(build_no_evidence_report(self.output_folder, self.config, fitted_paths))
                return

            compatible_by_job = [
                tuple(
                    model_id for model_id in self.model_ids
                    if self._model_reason(model_id, record) is None
                )
                for _spec, record in self.jobs
            ]
            evidence_splits = (
                self.config.screening_folds
                if self.config.strategy in {
                    EvaluationStrategy.NESTED_CV,
                    EvaluationStrategy.EXTERNAL_VALIDATION,
                }
                else 1
            )
            steps_by_job = [
                evaluation_run_count(model_ids, self.config, evidence_splits, record)
                for (_spec, record), model_ids in zip(self.jobs, compatible_by_job)
            ]
            task_count = (
                len({record.target_task.key for (_spec, record), models in zip(self.jobs, compatible_by_job) if models})
                if self.config.refit_on_all_data
                or self.config.strategy == EvaluationStrategy.EXTERNAL_VALIDATION
                else 0
            )
            screening_total = sum(steps_by_job)
            total = screening_total + task_count
            all_results: list[EvaluationResult] = []
            prepared_by_experiment: dict[str, tuple[np.ndarray, np.ndarray, DatasetIdentity]] = {}
            self.log_message.emit(
                "Run configuration: "
                f"scalers={self.config.selected_scalers}, losses={self.config.selected_losses}, "
                f"repetitions={self.config.repetitions}, validation={self.config.validation_ratio:.0%}, "
                f"model_multiplier={self.config.model_multiplier:g}, augmentation={self.config.augmentation_policy}:"
                f"{self.config.selected_augmentations or ('none',)}, "
                f"include_original={self.config.augmentation_include_original}, "
                f"foma=(alpha={self.config.foma_alpha:g}, k={self.config.foma_k})"
            )
            screening_offset = 0
            for (spec, record), compatible_models, job_steps in zip(
                self.jobs, compatible_by_job, steps_by_job
            ):
                if self.isInterruptionRequested():
                    self.cancelled.emit(tr("Run cancelled."))
                    return
                self.log_message.emit(f"[{spec.experiment_id}] {record.display_name}")
                for method in self.config.selected_augmentations:
                    augmentation_reason = augmentation_incompatibility_reason(method, record)
                    if augmentation_reason:
                        self._record_augmentation_skip(record, method, augmentation_reason)
                        self.log_message.emit(
                            f"  Skip augmentation {method} for this Experiment: "
                            f"{augmentation_reason}; other augmentation variants remain active"
                        )
                for model_id in self.model_ids:
                    reason = self._model_reason(model_id, record)
                    if reason:
                        self._record_skip(record, model_id, reason)
                        self.log_message.emit(f"  Skip {model_id}: {reason}")
                if not compatible_models:
                    self.log_message.emit("  No compatible selected models; experiment skipped")
                    continue
                X, y, identity, dropped = prepare_experiment_data(
                    self.bundle, spec, self.observation_ref, self.group_ref
                )
                prepared_by_experiment[record.stable_key] = (X, y, identity)
                if self.observation_ref is None:
                    self.log_message.emit(tr("  Warning: no Observation ID was named; row position is used as the alignment fallback."))
                self.log_message.emit(f"  rows={len(X)}, dropped={dropped}, grouped={identity.is_grouped}")
                def on_progress(state: ProgressState, experiment_id: str = spec.experiment_id) -> None:
                    if self.isInterruptionRequested():
                        raise InterruptedError
                    label = f"{experiment_id}: {record.display_name} | {state.stage_label}"
                    self.progress_changed.emit(state.current, state.total, label)

                results: list[EvaluationResult] = []
                model_offset = screening_offset
                for model_id in compatible_models:
                    model_steps = evaluation_run_count(
                        (model_id,), self.config, evidence_splits, record
                    )
                    try:
                        if self.config.strategy in {
                            EvaluationStrategy.NESTED_CV,
                            EvaluationStrategy.EXTERNAL_VALIDATION,
                        }:
                            model_results = evaluate_screening(
                                record,
                                X,
                                y,
                                identity,
                                (model_id,),
                                folds=self.config.screening_folds,
                                random_seed=self.config.random_seed,
                                progress_callback=on_progress,
                                progress_offset=model_offset,
                                progress_total=total,
                                config=self.config,
                            )
                        else:
                            model_results = evaluate_holdout(
                                record,
                                X,
                                y,
                                identity,
                                (model_id,),
                                test_size=self.config.test_size,
                                random_seed=self.config.random_seed,
                                progress_callback=on_progress,
                                progress_offset=model_offset,
                                progress_total=total,
                                config=self.config,
                            )
                        results.extend(model_results)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        self._record_skip(record, model_id, reason, stage="screening")
                        self.log_message.emit(
                            f"  Failed {model_id}; continuing remaining models: {reason}"
                        )
                    finally:
                        model_offset += model_steps
                all_results.extend(results)
                for result in results:
                    epoch_text = format_trained_epochs(
                        result.trained_epochs, self.config.epochs
                    )
                    epoch_suffix = f", {epoch_text}" if epoch_text else ""
                    self.log_message.emit(
                        f"  {result.model_name}: MAE={result.metrics.mae:.6g}, "
                        f"MAPE={result.metrics.mape_percent:.2f}%, "
                        f"R²={result.metrics.reported_r2:.4f}, diagnostic={result.metrics.diagnostic_r2:.4f}"
                        f"{epoch_suffix}"
                    )
                screening_offset += job_steps

            if not all_results:
                runtime_failures = [
                    row for row in self.failure_rows if row.get("stage") == "screening"
                ]
                if runtime_failures:
                    detail = "; ".join(
                        f"{row['model_id']}: {row['reason']}" for row in runtime_failures[:5]
                    )
                    raise ValueError(f"No selected model completed Screening. {detail}")
                raise ValueError("No selected model is compatible with the selected experiments")

            confirmation_results: list[ConfirmationResult] = []
            if self.config.run_confirmation:
                if self.config.strategy not in {
                    EvaluationStrategy.NESTED_CV,
                    EvaluationStrategy.EXTERNAL_VALIDATION,
                }:
                    raise ValueError("Confirmation requires development-data CV screening")
                promoted = promote_finalists(all_results, self.config.finalists_per_task)
                finalists = [finalist for rows in promoted.values() for finalist in rows]
                augmentation_methods = (
                    self.config.selected_augmentations
                    if (
                        self.config.confirmation_augmentation_ablation
                        and self.config.augmentation_policy != "original_only"
                    )
                    else ()
                )
                confirmation_config = ConfirmationConfig(
                    outer_folds=self.config.confirmation_folds,
                    outer_repeats=self.config.confirmation_repeats,
                    inner_folds=self.config.confirmation_inner_folds,
                    tuning_budget=self.config.confirmation_tuning_budget,
                    finalists_per_task=self.config.finalists_per_task,
                    augmentation_methods=tuple(augmentation_methods),
                    random_seed=self.config.random_seed,
                )
                self.log_message.emit(
                    f"Confirmation stage: {len(finalists)} finalist(s), "
                    f"outer={confirmation_config.outer_folds}x{confirmation_config.outer_repeats}, "
                    f"inner={confirmation_config.inner_folds}, budget={confirmation_config.tuning_budget}, "
                    f"augmentation={('original', *confirmation_config.augmentation_methods)}"
                )
                for finalist_index, finalist in enumerate(finalists, start=1):
                    if self.isInterruptionRequested():
                        raise InterruptedError
                    X, y, identity = prepared_by_experiment[finalist.experiment.stable_key]

                    def confirmation_progress(
                        state: ProgressState,
                        current_finalist: int = finalist_index,
                        finalist_count: int = len(finalists),
                    ) -> None:
                        if self.isInterruptionRequested():
                            raise InterruptedError
                        self.progress_changed.emit(
                            state.current,
                            state.total,
                            f"Confirmation finalist {current_finalist}/{finalist_count}: "
                            f"{state.experiment_label} | {state.stage_label}",
                        )

                    confirmed = confirm_finalist(
                        finalist,
                        X,
                        y,
                        identity,
                        confirmation_config,
                        self.config,
                        progress_callback=confirmation_progress,
                    )
                    confirmation_results.extend(confirmed)
                    for result in confirmed:
                        epoch_text = format_trained_epochs(
                            result.trained_epochs, self.config.epochs
                        )
                        epoch_suffix = f", {epoch_text}" if epoch_text else ""
                        self.log_message.emit(
                            f"  Confirmation {result.finalist.model_id} | {result.augmentation_id}: "
                            f"NMAE={result.nmae:.6g} "
                            f"(95% CI {result.nmae_ci_low:.6g}..{result.nmae_ci_high:.6g}), "
                            f"MAPE={result.metrics.mape_percent:.2f}%"
                            f"{epoch_suffix}"
                        )

            keep_limit = MODEL_KEEP_LIMIT
            if confirmation_results:
                winners = []
                for task_key in sorted({result.target_task_key for result in confirmation_results}):
                    ranked = sorted(
                        (r for r in confirmation_results if r.target_task_key == task_key),
                        key=lambda result: result.nmae,
                    )
                    kept = 0
                    for rank, result in enumerate(ranked, start=1):
                        reported = result.metrics.reported_r2
                        if not np.isfinite(reported) or reported <= 0.0:
                            self.log_message.emit(
                                f"Not saved ({task_key} confirmation rank {rank}): "
                                f"{result.finalist.model_id} | {result.augmentation_id} — "
                                f"Reported R²={reported:.4f} is not above 0"
                            )
                            continue
                        if kept >= keep_limit:
                            break
                        kept += 1
                        winners.append(
                            (
                                result.finalist.screening_result,
                                result.augmentation_id,
                                representative_parameters(result),
                                kept,
                                "confirmation",
                                result.metrics,
                            )
                        )
            else:
                retained, dropped = select_retained_results(all_results, limit=keep_limit)
                for result, reason in dropped:
                    self.log_message.emit(
                        f"Not saved ({result.target_task_key}): {result.model_id} | "
                        f"{result.loss_id} | {result.scaler_id} | {result.augmentation_id} — {reason}"
                    )
                winners = [
                    (result, result.augmentation_id, {}, rank, "screening", result.metrics)
                    for rows in retained.values()
                    for rank, result in rows
                ]
            if not winners:
                self.log_message.emit(
                    "No model reached Reported R² > 0, so no deployable bundle was saved for any Target Task."
                )

            external_results: list[EvaluationResult] = []
            external_bundles = []
            if self.config.strategy == EvaluationStrategy.EXTERNAL_VALIDATION:
                assert self.external_bundle is not None
                self.log_message.emit(
                    "External validation: development selection is frozen; external targets are now opened once."
                )
                for winner_index, (
                    winner,
                    selected_augmentation,
                    tuned_parameters,
                    winner_rank,
                    evidence_stage,
                    winner_metrics,
                ) in enumerate(winners):
                    spec, record = next(
                        job for job in self.jobs if job[1].stable_key == winner.experiment.stable_key
                    )
                    development_X, development_y, _identity = prepared_by_experiment[record.stable_key]
                    external_X, external_y, external_identity, dropped = prepare_external_experiment_data(
                        self.external_bundle,
                        spec,
                        self.external_mapping,
                        self.external_observation_ref,
                    )
                    self.log_message.emit(
                        f"[{spec.experiment_id}] external rows={len(external_X)}, dropped={dropped}"
                    )

                    def external_progress(state: ProgressState, experiment_id: str = spec.experiment_id) -> None:
                        if self.isInterruptionRequested():
                            raise InterruptedError
                        self.progress_changed.emit(
                            state.current,
                            state.total,
                            f"{experiment_id}: {record.display_name} | external validation",
                        )

                    external_result, fitted_bundle = evaluate_external_model(
                        winner,
                        development_X,
                        development_y,
                        external_X,
                        external_y,
                        external_identity,
                        random_seed=self.config.random_seed,
                        config=self.config,
                        model_parameters=tuned_parameters,
                        selected_augmentation_id=selected_augmentation,
                        augmentation_methods=augmentation_methods_from_id(selected_augmentation),
                        progress_callback=external_progress,
                        progress_offset=screening_total + winner_index,
                        progress_total=total,
                    )
                    external_results.append(external_result)
                    external_bundles.append(fitted_bundle)
                    epoch_text = format_trained_epochs(
                        external_result.trained_epochs, self.config.epochs
                    )
                    epoch_suffix = f", {epoch_text}" if epoch_text else ""
                    self.log_message.emit(
                        f"  External {external_result.model_name}: MAE={external_result.metrics.mae:.6g}, "
                        f"MAPE={external_result.metrics.mape_percent:.2f}%, "
                        f"R²={external_result.metrics.reported_r2:.4f}, "
                        f"diagnostic={external_result.metrics.diagnostic_r2:.4f}"
                        f"{epoch_suffix}"
                    )

            if self.config.refit_on_all_data:
                fitted_paths: list[Path] = []
                if external_bundles:
                    fitted_paths.extend(
                        save_fitted_models(
                            external_bundles,
                            self.output_folder,
                            [
                                bundle_metadata(
                                    bundle,
                                    rank=rank,
                                    evidence_stage="external_validation",
                                    metrics=result.metrics,
                                    extra={"run_label": self.run_label},
                                )
                                for bundle, result, rank in zip(
                                    external_bundles,
                                    external_results,
                                    [winner[3] for winner in winners],
                                )
                            ],
                        )
                    )
                else:
                    refit_offset = screening_total
                    for winner_index, (
                        winner,
                        selected_augmentation,
                        tuned_parameters,
                        winner_rank,
                        evidence_stage,
                        winner_metrics,
                    ) in enumerate(winners):
                        spec, record = next(
                            job for job in self.jobs if job[1].stable_key == winner.experiment.stable_key
                        )
                        X, y, _identity, _dropped = prepare_experiment_data(
                            self.bundle, spec, self.observation_ref, self.group_ref
                        )

                        def refit_progress(state: ProgressState, experiment_id: str = spec.experiment_id) -> None:
                            if self.isInterruptionRequested():
                                raise InterruptedError
                            self.progress_changed.emit(
                                state.current,
                                state.total,
                                f"{experiment_id}: {record.display_name} | {state.stage_label}",
                            )

                        final_parameters = dict(tuned_parameters)
                        final_config = replace(
                            self.config,
                            repetitions=1,
                            selected_scalers=(winner.scaler_id,),
                            selected_losses=((winner.loss_id,) if winner.loss_id != "native" else ("mse",)),
                            augmentation_policy="original_only",
                            selected_augmentations=(),
                            learning_rate=float(final_parameters.pop("learning_rate", self.config.learning_rate)),
                            model_multiplier=float(final_parameters.pop("model_multiplier", self.config.model_multiplier)),
                        )
                        bundles = fit_final_models(
                            record,
                            X,
                            y,
                            (winner.model_id,),
                            random_seed=self.config.random_seed,
                            evidence_note="Selected from evaluation, then refit on all available rows; final-fit score is not evidence.",
                            progress_callback=refit_progress,
                            progress_offset=refit_offset + winner_index,
                            progress_total=total,
                            config=final_config,
                            model_parameters=final_parameters,
                            augmentation_override=(
                                selected_augmentation,
                                augmentation_methods_from_id(selected_augmentation),
                            ),
                        )
                        for bundle in bundles:
                            epoch_text = format_trained_epochs(
                                (bundle.trained_epochs,), self.config.epochs
                            )
                            if epoch_text:
                                self.log_message.emit(
                                    f"  {bundle.model_name} final fit: {epoch_text}"
                                )
                        self.log_message.emit(
                            f"  Keeping rank {winner_rank}/{keep_limit} of "
                            f"{record.target_task.key}: {winner.model_name} "
                            f"(R²={winner_metrics.reported_r2:.4f}, "
                            f"MAPE={winner_metrics.mape_percent:.2f}%)"
                        )
                        fitted_paths.extend(
                            save_fitted_models(
                                bundles,
                                self.output_folder,
                                [
                                    bundle_metadata(
                                        bundle,
                                        rank=winner_rank,
                                        evidence_stage=evidence_stage,
                                        metrics=winner_metrics,
                                        extra={"run_label": self.run_label},
                                    )
                                    for bundle in bundles
                                ],
                            )
                        )
                self.log_message.emit(
                    f"Saved {len(fitted_paths)} final model bundle(s) under models/ "
                    f"(top {keep_limit} per Target Task, Reported R² > 0 only)."
                )
            artifacts = MasterReportBuilder(
                all_results,
                self.config,
                confirmation_results,
                external_results,
                self.bundle.audit_dataframe().to_dict(orient="records"),
                self._experiment_audit_rows(),
                self.failure_rows,
            ).build(self.output_folder)
            self.completed.emit(artifacts)
        except InterruptedError:
            self.cancelled.emit(tr("Run cancelled."))
        except Exception as exc:
            import traceback

            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class TrainingPage(QWidget):
    """The Benchmark Run workspace, embeddable as one tab of the merged UI."""

    run_completed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bundle: DatasetBundle | None = None
        self.external_bundle: DatasetBundle | None = None
        self.legacy_specs: list[Any] = []
        self.catalog = ExperimentCatalog()
        self.worker: BenchmarkWorker | None = None
        self.sheet_role_boxes: dict[str, QComboBox] = {}
        self.column_role_boxes: dict[tuple[str, str], QComboBox] = {}
        self.model_checks: dict[str, QCheckBox] = {}
        self.scaler_checks: dict[str, QCheckBox] = {}
        self.loss_checks: dict[str, QCheckBox] = {}
        self.augmentation_checks: dict[str, QCheckBox] = {}
        self.external_mapping_boxes: dict[str, QComboBox] = {}
        self.observation_ref: ColumnRef | None = None
        self.group_ref: ColumnRef | None = None
        self.selection_memory: dict[str, bool] = {}
        self.structure_memory: dict[str, tuple[FeatureStructure, bool]] = {}
        self.training_log_path: Path | None = None
        self.current_run_output_folder: Path | None = None
        self.dataset_path: str | None = None
        self._role_change_guard = False
        self.setting_help_labels: list[QLabel] = []
        self.setting_help_controls: list[QWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), tr("1. Data and combinations"))
        self.tabs.addTab(self._build_config_tab(), tr("2. Evaluation settings"))
        self.tabs.addTab(self._build_run_tab(), tr("3. Run and report"))
        layout.addWidget(self.tabs)

    def _build_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        drop = DatasetDropFrame()
        drop.file_dropped.connect(self.load_dataset)
        layout.addWidget(drop)
        row = QHBoxLayout()
        browse = QPushButton(tr("Choose a data file"))
        browse.clicked.connect(self.browse_dataset)
        self.dataset_label = QLabel(tr("No data loaded"))
        preview = QPushButton(tr("Generate / refresh combinations"))
        preview.clicked.connect(self.generate_preview)
        row.addWidget(browse)
        row.addWidget(self.dataset_label, 1)
        row.addWidget(preview)
        layout.addLayout(row)
        self.dataset_summary = QLabel(
            tr("Observation ID and Group ID never become model features.")
        )
        self.dataset_summary.setWordWrap(True)
        layout.addWidget(self.dataset_summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.role_tree = QTreeWidget()
        self.role_tree.setColumnCount(6)
        self.role_tree.setHeaderLabels(
            (
                tr("Sheet / column"),
                tr("Role"),
                tr("Type"),
                tr("Rows"),
                tr("Missing"),
                tr("Audit"),
            )
        )
        splitter.addWidget(self.role_tree)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        tools = QHBoxLayout()
        self.combo_label = QLabel(tr("No combinations generated yet"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            tr("Filter by ID, Inputs, Outputs or Stable Key…")
        )
        all_button = QPushButton(tr("Select all"))
        none_button = QPushButton(tr("Select none"))
        invert_button = QPushButton(tr("Invert"))
        confirm_structures_button = QPushButton(
            tr("Confirm the suggested structure for the selected combinations")
        )
        tools.addWidget(self.combo_label)
        tools.addWidget(self.filter_edit, 1)
        tools.addWidget(all_button)
        tools.addWidget(none_button)
        tools.addWidget(invert_button)
        tools.addWidget(confirm_structures_button)
        right_layout.addLayout(tools)
        self.table_model = ExperimentTableModel(self.catalog)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.table_model)
        self.proxy_model.setFilterKeyColumn(-1)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.filter_edit.textChanged.connect(self.proxy_model.setFilterFixedString)
        self.combo_table = QTableView()
        self.combo_table.setModel(self.proxy_model)
        self.combo_table.setItemDelegateForColumn(9, FeatureStructureDelegate(self.combo_table))
        self.combo_table.setSortingEnabled(True)
        self.combo_table.setAlternatingRowColors(True)
        self.combo_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.combo_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.combo_table.setColumnWidth(0, 54)
        self.combo_table.setColumnWidth(1, 100)
        self.combo_table.setColumnWidth(2, 340)
        self.combo_table.setColumnWidth(3, 280)
        self.combo_table.setColumnWidth(7, 190)
        self.combo_table.setColumnWidth(8, 190)
        self.combo_table.setColumnWidth(9, 170)
        self.combo_table.setColumnWidth(10, 90)
        right_layout.addWidget(self.combo_table)
        all_button.clicked.connect(lambda: self._set_selection("all"))
        none_button.clicked.connect(lambda: self._set_selection("none"))
        invert_button.clicked.connect(lambda: self._set_selection("invert"))
        confirm_structures_button.clicked.connect(self._confirm_selected_structures)
        self.table_model.dataChanged.connect(self._update_selection_label)
        splitter.addWidget(right)
        splitter.setSizes((700, 820))
        layout.addWidget(splitter, 1)
        return tab

    def _help_label(self, text: str, explanation: str) -> QLabel:
        label = QLabel(f"{text} {HELP_GLYPH}")
        label.setToolTip(explanation)
        self.setting_help_labels.append(label)
        return label

    def _help_control(self, control: QWidget, explanation: str, *, append_question: bool = False) -> QWidget:
        control.setToolTip(explanation)
        if append_question and hasattr(control, "text") and hasattr(control, "setText"):
            current = control.text()  # type: ignore[attr-defined]
            if not current.rstrip().endswith(HELP_GLYPH):
                control.setText(f"{current} {HELP_GLYPH}")  # type: ignore[attr-defined]
        self.setting_help_controls.append(control)
        return control

    def _add_help_row(
        self,
        form: QFormLayout,
        text: str,
        control: QWidget,
        explanation: str,
    ) -> None:
        self._help_control(control, explanation)
        form.addRow(self._help_label(text, explanation), control)

    def _build_config_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)
        evaluation = QGroupBox(f"Evaluation Strategy {HELP_GLYPH}")
        evaluation.setToolTip(
            tr(
                "Decides how generalization evidence is produced. This is not a "
                "parameter preset: switching strategy only disables the fields that do "
                "not apply, and keeps the values you entered."
            )
        )
        form = QFormLayout(evaluation)
        self.strategy_combo = QComboBox()
        strategy_rows = (
            (
                tr("Two-stage benchmark: CV Screening + optional Confirmation"),
                EvaluationStrategy.NESTED_CV,
                tr(
                    "K-fold Screening on the development data, with optional repeated "
                    "nested Confirmation. Test ratio and External mapping are ignored."
                ),
            ),
            (
                tr("Holdout: one Train/Test split"),
                EvaluationStrategy.HOLDOUT,
                tr(
                    "Splits off one test set at random according to Test ratio. "
                    "Screening folds, Confirmation and External mapping are ignored."
                ),
            ),
            (
                tr("External Validation: an independent second dataset"),
                EvaluationStrategy.EXTERNAL_VALIDATION,
                tr(
                    "Selects the configuration with development CV, then evaluates once "
                    "against the mapped external data. Test ratio is ignored."
                ),
            ),
            (
                tr("Train Final Model Only: train, do not evaluate"),
                EvaluationStrategy.TRAIN_FINAL_ONLY,
                tr(
                    "Builds the final model straight from every development row. Test "
                    "ratio, folds, Confirmation and External mapping are ignored, and no "
                    "generalization metric is produced."
                ),
            ),
        )
        for label, strategy, explanation in strategy_rows:
            self.strategy_combo.addItem(label, strategy)
            self.strategy_combo.setItemData(
                self.strategy_combo.count() - 1,
                explanation,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._help_control(
            self.strategy_combo,
            tr(
                "Evaluation Strategy picks the evidence workflow. Unlike Run preset it "
                "never rewrites values such as epochs; settings that do not apply are "
                "disabled and ignored at run time."
            ),
        )
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 50.0)
        self.test_spin.setSuffix(" %")
        self.test_spin.setValue(20.0)
        self.test_spin.setSingleStep(5.0)
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 10)
        self.folds_spin.setValue(3)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.refit_check = QCheckBox(
            tr("After evaluation, refit the final model on all available data")
        )
        self.refit_check.setChecked(True)
        self.run_confirmation_check = QCheckBox(
            tr("Run repeated nested Confirmation after Screening")
        )
        self.run_confirmation_check.setChecked(False)
        self.finalists_spin = QSpinBox(); self.finalists_spin.setRange(1, 20); self.finalists_spin.setValue(5)
        self.confirmation_outer_folds_spin = QSpinBox(); self.confirmation_outer_folds_spin.setRange(2, 10); self.confirmation_outer_folds_spin.setValue(5)
        self.confirmation_outer_repeats_spin = QSpinBox(); self.confirmation_outer_repeats_spin.setRange(1, 20); self.confirmation_outer_repeats_spin.setValue(3)
        self.confirmation_inner_folds_spin = QSpinBox(); self.confirmation_inner_folds_spin.setRange(2, 10); self.confirmation_inner_folds_spin.setValue(3)
        self.tuning_budget_spin = QSpinBox(); self.tuning_budget_spin.setRange(1, 100); self.tuning_budget_spin.setValue(8)
        self.confirmation_aug_check = QCheckBox(
            tr("Compare Original against the selected augmentation on identical outer folds")
        )
        self.confirmation_aug_check.setChecked(True)
        self._add_help_row(
            form, tr("Strategy"), self.strategy_combo,
            tr(
                "Chooses how generalization evidence is produced. It is not a preset: it "
                "never rewrites the values below, it only uses or ignores the matching "
                "fields."
            ),
        )
        self._add_help_row(
            form, tr("Test ratio"), self.test_spin,
            tr(
                "Holdout only: splits off one test set by percentage. At 0% every row "
                "goes into the final fit, and no test evidence is produced."
            ),
        )
        self._add_help_row(
            form, tr("Screening folds"), self.folds_spin,
            tr(
                "Cuts the development data into K parts, evaluating on one and training "
                "on the rest in turn. A larger K means more evaluations and more time."
            ),
        )
        self._add_help_row(
            form, tr("Random seed"), self.seed_spin,
            tr(
                "Controls the random sequence for splitting, initialization and "
                "augmentation. The same data, settings and seed reproduce the result."
            ),
        )
        self._help_control(
            self.refit_check,
            tr(
                "Once evaluation has chosen a configuration, retrain a deployable final "
                "model on every development row. That fit is not generalization evidence."
            ),
            append_question=True,
        )
        form.addRow(self.refit_check)
        self._help_control(
            self.run_confirmation_check,
            tr(
                "Re-runs the candidates promoted from Screening through repeated nested "
                "CV. More reliable, substantially more compute, CV strategies only."
            ),
            append_question=True,
        )
        form.addRow(self.run_confirmation_check)
        self._add_help_row(
            form, tr("Finalists / Target Task"), self.finalists_spin,
            tr(
                "How many candidate configurations per identical output target are "
                "promoted to Confirmation. More finalists means a longer Confirmation."
            ),
        )
        self._add_help_row(
            form, tr("Confirmation outer folds"), self.confirmation_outer_folds_spin,
            tr(
                "How many splits the outer Confirmation CV uses. Every outer test fold "
                "stays entirely out of that round's tuning and training."
            ),
        )
        self._add_help_row(
            form, tr("Confirmation outer repeats"), self.confirmation_outer_repeats_spin,
            tr(
                "Repeats the whole outer CV with different seeds, which estimates the "
                "variance of the result and multiplies the compute accordingly."
            ),
        )
        self._add_help_row(
            form, tr("Confirmation inner folds"), self.confirmation_inner_folds_spin,
            tr(
                "Inner CV runs only inside each outer training partition, so parameters "
                "are never tuned against outer test data."
            ),
        )
        self._add_help_row(
            form, tr("Tuning budget / finalist"), self.tuning_budget_spin,
            tr(
                "How many parameter sets each finalist may try per outer fold. A larger "
                "budget searches wider and takes longer."
            ),
        )
        self._help_control(
            self.confirmation_aug_check,
            tr(
                "Compares Original against the enabled augmentation on exactly the same "
                "outer folds. Ignored under Original only."
            ),
            append_question=True,
        )
        form.addRow(self.confirmation_aug_check)
        self.strategy_description_label = QLabel()
        self.strategy_description_label.setWordWrap(True)
        self.strategy_description_label.setStyleSheet("color: #315a7d; padding: 4px;")
        form.addRow(self.strategy_description_label)
        note = QLabel(
            tr(
                "Note: a greyed-out field is one the current strategy does not use. "
                "Switching strategy keeps whatever you typed."
            )
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.confirmation_controls = (
            self.finalists_spin,
            self.confirmation_outer_folds_spin,
            self.confirmation_outer_repeats_spin,
            self.confirmation_inner_folds_spin,
            self.tuning_budget_spin,
            self.confirmation_aug_check,
        )
        self.strategy_combo.currentIndexChanged.connect(self._update_strategy_controls)
        self.run_confirmation_check.toggled.connect(self._update_strategy_controls)
        layout.addWidget(evaluation, 0, 0)

        models_box = QGroupBox(
            tr("Models (greyed-out entries are kept on purpose, so none is forgotten)")
            + f" {HELP_GLYPH}"
        )
        models_box.setToolTip(
            tr(
                "Tick the models to compare. Each extra model expands into more training "
                "runs across the selected scalers, losses, augmentations and repetitions."
            )
        )
        models_layout = QGridLayout(models_box)
        helps = model_help()
        for index, capability in enumerate(model_registry()):
            check = QCheckBox(capability.display_name)
            check.setEnabled(capability.enabled)
            check.setChecked(capability.model_id in {"mean", "ridge", "pls", "svr_rbf"})
            if capability.availability == ModelAvailability.COMING_LATER:
                check.setText(f"{capability.display_name} — Coming Later")
            elif not capability.enabled:
                check.setText(f"{capability.display_name} — Inactive")
            model_explanation = helps.get(
                capability.model_id,
                tr(
                    "Ticking this adds the model to every compatible Experiment. More "
                    "models means more total run time."
                ),
            )
            if not capability.enabled:
                model_explanation += " " + tr(
                    "Currently unavailable: {reason}", reason=capability.reason
                )
            self._help_control(check, model_explanation, append_question=True)
            self.model_checks[capability.model_id] = check
            models_layout.addWidget(check, index // 2, index % 2)
        layout.addWidget(models_box, 0, 1)

        advanced = QGroupBox(
            tr("Advanced training settings (a preset only fills in defaults; every field stays editable)")
            + f" {HELP_GLYPH}"
        )
        advanced.setToolTip(
            tr(
                "Controls neural training, repetitions and feature scaling. Run preset is "
                "the only preset that immediately rewrites some of these values."
            )
        )
        advanced_form = QFormLayout(advanced)
        self.run_preset_combo = QComboBox()
        self.run_preset_combo.addItem("Economy", "economy")
        self.run_preset_combo.addItem("Balanced", "balanced")
        self.run_preset_combo.addItem("Rigorous", "rigorous")
        self.run_preset_combo.addItem("Custom", "custom")
        preset_help = {
            "economy": tr(
                "Quick look: fewer epochs, less patience and fewer repetitions. Saves "
                "time, with less stable results."
            ),
            "balanced": tr(
                "Balanced: moderate epochs and patience with 3 repetitions, trading time "
                "against stability."
            ),
            "rigorous": tr(
                "Rigorous: more epochs, more patience, wider models and 5 repetitions. "
                "Substantially more compute."
            ),
            "custom": tr("Custom: rewrites nothing and keeps your current settings."),
        }
        for index in range(self.run_preset_combo.count()):
            self.run_preset_combo.setItemData(
                index,
                preset_help[self.run_preset_combo.itemData(index)],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1_000_000); self.epochs_spin.setValue(500)
        self.patience_spin = QSpinBox(); self.patience_spin.setRange(1, 100_000); self.patience_spin.setValue(30)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(2, 100_000); self.batch_spin.setValue(32)
        self.learning_rate_spin = QDoubleSpinBox(); self.learning_rate_spin.setDecimals(7); self.learning_rate_spin.setRange(1e-7, 1.0); self.learning_rate_spin.setValue(0.001)
        self.model_multiplier_spin = QDoubleSpinBox(); self.model_multiplier_spin.setRange(0.10, 20.0); self.model_multiplier_spin.setSingleStep(0.25); self.model_multiplier_spin.setValue(1.0)
        self.repetitions_spin = QSpinBox(); self.repetitions_spin.setRange(1, 1000); self.repetitions_spin.setValue(1)
        self.validation_spin = QDoubleSpinBox(); self.validation_spin.setRange(0.05, 0.50); self.validation_spin.setSingleStep(0.05); self.validation_spin.setValue(0.20)
        scaler_row = QWidget(); scaler_layout = QHBoxLayout(scaler_row); scaler_layout.setContentsMargins(0, 0, 0, 0)
        scaler_help = {
            "standard": tr(
                "Standard scaler: uses the training fold's mean and standard deviation to "
                "reach roughly zero mean and unit scale. Suits most models."
            ),
            "robust": tr(
                "Robust scaler: scales by the median and interquartile range, so outliers "
                "matter less."
            ),
            "minmax": tr(
                "MinMax scaler: scales into a fixed range using the training fold's "
                "minimum and maximum. More sensitive to extreme values."
            ),
        }
        for scaler_id, label in (("standard", "Standard"), ("robust", "Robust"), ("minmax", "MinMax")):
            check = QCheckBox(label)
            check.setChecked(scaler_id == "standard")
            self._help_control(check, scaler_help[scaler_id], append_question=True)
            self.scaler_checks[scaler_id] = check
            scaler_layout.addWidget(check)
        self._add_help_row(
            advanced_form, tr("Run preset"), self.run_preset_combo,
            tr(
                "The one real parameter preset: choosing Economy, Balanced or Rigorous "
                "immediately rewrites epochs, patience, repetitions, validation ratio and "
                "model multiplier. Nothing else changes."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Epochs"), self.epochs_spin,
            tr(
                "How many complete passes over the training data a neural model may make. "
                "More can learn more, at the cost of time and overfitting risk."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Early-stopping patience"), self.patience_spin,
            tr(
                "How many epochs without validation-loss improvement to allow before "
                "stopping. Larger waits longer; smaller may stop too early."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Batch size"), self.batch_spin,
            tr(
                "How many training rows each gradient update uses. Larger is usually "
                "faster but needs more memory; smaller makes updates noisier."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Learning rate"), self.learning_rate_spin,
            tr(
                "The step size the neural optimizer takes per weight update. Too large is "
                "unstable; too small converges slowly."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Model width multiplier"), self.model_multiplier_spin,
            tr(
                "Scales the neural hidden width up or down. Larger raises capacity, "
                "memory use and overfitting risk together."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Training repetitions"), self.repetitions_spin,
            tr(
                "Repeats every configuration with a different random initialization and "
                "pools the predictions. Reduces luck, multiplies the time."
            ),
        )
        self._add_help_row(
            advanced_form, tr("Internal validation ratio"), self.validation_spin,
            tr(
                "Carves an early-stopping watch set out of the neural model's "
                "fold-training partition only. Fold test data is never touched."
            ),
        )
        self._help_control(
            scaler_row,
            tr(
                "Tick one or more fold-local scalings. Each extra scaler adds a whole set "
                "of model configurations."
            ),
        )
        advanced_form.addRow(
            self._help_label(
                tr("Scalers"),
                tr(
                    "Controls how X and y are scaled inside each training fold. Every "
                    "ticked scaler is trained and evaluated separately."
                ),
            ),
            scaler_row,
        )
        self.run_preset_combo.currentIndexChanged.connect(self._apply_run_preset)
        layout.addWidget(advanced, 1, 0)

        losses = QGroupBox(
            tr("Loss functions (only for neural adapters that accept a custom loss)")
            + f" {HELP_GLYPH}"
        )
        losses.setToolTip(
            tr(
                "The loss decides how a neural model measures prediction error while "
                "training. Each extra tick adds one neural configuration."
            )
        )
        loss_layout = QVBoxLayout(losses)
        loss_texts = loss_help()
        for loss_id, label, checked in (
            ("mse", "MSE", True),
            ("huber", tr("Huber / SmoothL1 (suggested for noisy measurements)"), True),
            ("mae", "MAE / L1", False),
            ("log_cosh", "Log-cosh", False),
        ):
            check = QCheckBox(label)
            check.setChecked(checked)
            self._help_control(check, loss_texts[loss_id], append_question=True)
            self.loss_checks[loss_id] = check
            loss_layout.addWidget(check)
        loss_note = QLabel(
            tr(
                "Classical models use their own native objective, so ticking several "
                "losses does not run them more than once."
            )
        )
        loss_note.setWordWrap(True); loss_layout.addWidget(loss_note)
        layout.addWidget(losses, 1, 1)

        augmentation = CollapsibleSection(f"Augmentation policy {HELP_GLYPH}", expanded=False)
        augmentation.toggle_button.setToolTip(
            tr(
                "Decides how synthetic training rows enter the comparison. Original only "
                "ignores every ticked method."
            )
        )
        self.augmentation_section = augmentation
        aug_layout = QFormLayout()
        augmentation.setContentLayout(aug_layout)
        self.augmentation_policy_combo = QComboBox()
        self.augmentation_policy_combo.addItem(
            tr("Original only (no augmentation)"), "original_only"
        )
        self.augmentation_policy_combo.addItem(
            tr("Compare separately against the original data (suggested)"), "compare_separately"
        )
        self.augmentation_policy_combo.addItem(tr("Combine the ticked methods"), "combined")
        policy_help = {
            "original_only": tr(
                "Builds the Original configuration only. Your method ticks are kept but "
                "completely ignored at run time."
            ),
            "compare_separately": tr(
                "Builds Original plus one separate configuration per ticked method, so you "
                "can see directly whether each method improves the result."
            ),
            "combined": tr(
                "Builds Original plus one configuration combining every ticked method. "
                "Tests the combination, but cannot attribute the effect to one method."
            ),
        }
        for index in range(self.augmentation_policy_combo.count()):
            self.augmentation_policy_combo.setItemData(
                index,
                policy_help[self.augmentation_policy_combo.itemData(index)],
                Qt.ItemDataRole.ToolTipRole,
            )
        self._add_help_row(
            aug_layout, tr("Policy"), self.augmentation_policy_combo,
            tr(
                "Controls how augmentation configurations expand: Original only skips "
                "augmentation, Separate compares each method on its own, Combined applies "
                "them together."
            ),
        )
        method_widget = QWidget(); method_layout = QGridLayout(method_widget); method_layout.setContentsMargins(0, 0, 0, 0)
        methods = (
            ("c_mixup", tr("C-Mixup (target-aware)")),
            ("calibrated_noise", tr("Calibrated feature noise")),
            ("spectral_perturbation", tr("Spectral perturbation (needs a valid wavelength axis)")),
            ("feature_masking", tr("Training-time feature masking")),
            ("foma", "FOMA (First-Order Manifold Data Augmentation)"),
        )
        augmentation_texts = augmentation_help()
        for index, (method_id, label) in enumerate(methods):
            check = QCheckBox(label)
            self._help_control(check, augmentation_texts[method_id], append_question=True)
            self.augmentation_checks[method_id] = check
            method_layout.addWidget(check, index // 2, index % 2)
        self._help_control(
            method_widget,
            tr(
                "Choose which training-only augmentations to generate. Policy decides "
                "whether they are compared separately or combined."
            ),
        )
        aug_layout.addRow(
            self._help_label(
                tr("Methods"),
                tr(
                    "Tick augmentation algorithms. Under Original only they are all "
                    "ignored; under Separate each is compared on its own; under Combined "
                    "they are applied together."
                ),
            ),
            method_widget,
        )
        self.augmentation_repeats_spin = QSpinBox(); self.augmentation_repeats_spin.setRange(1, 100); self.augmentation_repeats_spin.setValue(1)
        self.augmentation_ratio_spin = QDoubleSpinBox(); self.augmentation_ratio_spin.setRange(0.05, 5.0); self.augmentation_ratio_spin.setSingleStep(0.25); self.augmentation_ratio_spin.setValue(0.50)
        self.augmentation_include_original_check = QCheckBox(
            tr("Include the original rows in the augmented training set")
        )
        self.augmentation_include_original_check.setChecked(True)
        self.c_mixup_alpha_spin = QDoubleSpinBox(); self.c_mixup_alpha_spin.setRange(0.05, 20.0); self.c_mixup_alpha_spin.setValue(2.0)
        self.noise_fraction_spin = QDoubleSpinBox(); self.noise_fraction_spin.setDecimals(4); self.noise_fraction_spin.setRange(0.0, 1.0); self.noise_fraction_spin.setValue(0.02)
        self.feature_mask_probability_spin = QDoubleSpinBox(); self.feature_mask_probability_spin.setDecimals(3); self.feature_mask_probability_spin.setRange(0.0, 0.95); self.feature_mask_probability_spin.setValue(0.05)
        self.foma_alpha_spin = QDoubleSpinBox(); self.foma_alpha_spin.setRange(0.05, 20.0); self.foma_alpha_spin.setValue(2.0)
        self.foma_k_spin = QSpinBox(); self.foma_k_spin.setRange(1, 10_000); self.foma_k_spin.setValue(1)
        self._add_help_row(
            aug_layout, tr("Synthetic rows / train rows"), self.augmentation_ratio_spin,
            tr(
                "How many synthetic rows each augmentation produces relative to the "
                "fold-training rows. 0.5 means about half as many."
            ),
        )
        self._add_help_row(
            aug_layout, tr("Augmentation repeats"), self.augmentation_repeats_spin,
            tr(
                "Multiplies the synthetic row count again. More data and more training "
                "time — this is not the same as model training repetitions."
            ),
        )
        self._help_control(
            self.augmentation_include_original_check,
            tr(
                "When ticked, an augmented variant trains on the original rows plus the "
                "synthetic ones. When cleared, that variant uses synthetic rows only; the "
                "Original baseline is kept either way."
            ),
            append_question=True,
        )
        aug_layout.addRow(self.augmentation_include_original_check)
        self._add_help_row(
            aug_layout, tr("C-Mixup alpha"), self.c_mixup_alpha_spin,
            tr(
                "The Beta distribution shape parameter controlling the interpolation "
                "weight between two rows. Larger mixes closer to the middle; smaller stays "
                "closer to one of them."
            ),
        )
        self._add_help_row(
            aug_layout, tr("Noise fraction"), self.noise_fraction_spin,
            tr(
                "Calibrated noise as a fraction of the training feature's standard "
                "deviation. Larger perturbs more, and is more likely to leave the "
                "plausible measurement range."
            ),
        )
        self._add_help_row(
            aug_layout, tr("Mask probability"), self.feature_mask_probability_spin,
            tr(
                "The chance that each training feature is masked to its scaled mean. "
                "Larger regularizes harder, and can discard too much information."
            ),
        )
        self._add_help_row(
            aug_layout, tr("FOMA alpha"), self.foma_alpha_spin,
            tr(
                "The Beta scaling coefficient FOMA applies to non-leading singular "
                "components. Larger favours moderate scaling; smaller more often either "
                "preserves or strongly compresses them."
            ),
        )
        self._add_help_row(
            aug_layout, tr("FOMA retained components (k)"), self.foma_k_spin,
            tr(
                "How many leading joint X/y SVD components FOMA keeps intact. A larger k "
                "preserves more of the data's main structure and perturbs more "
                "conservatively."
            ),
        )
        self.augmentation_policy_description_label = QLabel()
        self.augmentation_policy_description_label.setWordWrap(True)
        self.augmentation_policy_description_label.setStyleSheet("color: #315a7d; padding: 4px;")
        aug_layout.addRow(self.augmentation_policy_description_label)
        aug_note = QLabel(
            tr(
                "Every method acts only on the current fold's training partition, and the "
                "Original baseline is always kept. Bootstrap is not counted as "
                "augmentation; SMOGN needs a target relevance definition first, so this "
                "version offers no blind switch for it."
            )
        )
        aug_note.setWordWrap(True)
        aug_layout.addRow(aug_note)
        self.augmentation_detail_controls = (
            *self.augmentation_checks.values(),
            self.augmentation_ratio_spin,
            self.augmentation_repeats_spin,
            self.augmentation_include_original_check,
            self.c_mixup_alpha_spin,
            self.noise_fraction_spin,
            self.feature_mask_probability_spin,
            self.foma_alpha_spin,
            self.foma_k_spin,
        )
        self.augmentation_policy_combo.currentIndexChanged.connect(
            self._update_augmentation_controls
        )
        layout.addWidget(augmentation, 2, 0, 1, 2)

        external = CollapsibleSection(f"External Validation dataset mapping {HELP_GLYPH}", expanded=False)
        external.toggle_button.setToolTip(
            tr(
                "For the External Validation strategy only: map each development column to "
                "the independent external data. The external target is read only after the "
                "model has been selected."
            )
        )
        self.external_validation_section = external
        external_layout = QVBoxLayout()
        external.setContentLayout(external_layout)
        external_top = QHBoxLayout()
        external_button = QPushButton(tr("Load independent external data") + f" {HELP_GLYPH}")
        self._help_control(
            external_button,
            tr(
                "Loads a fully independent file that took no part in development "
                "selection. Used for exactly one final evaluation under the External "
                "Validation strategy."
            ),
        )
        external_button.clicked.connect(self.browse_external_dataset)
        self.external_dataset_label = QLabel(
            tr("Not loaded; used by the External Validation strategy only")
        )
        external_top.addWidget(external_button)
        external_top.addWidget(self.external_dataset_label, 1)
        external_layout.addLayout(external_top)
        self.external_observation_combo = QComboBox()
        self.external_observation_combo.addItem(tr("Use the external row position"), None)
        external_layout.addWidget(self._help_label(
            tr("External Observation ID (optional)"),
            tr(
                "The column that uniquely identifies each external observation. Without "
                "one, rows are aligned by position and the report says so."
            ),
        ))
        self._help_control(
            self.external_observation_combo,
            tr(
                "Names the external observation identifier. It is used for tracking and "
                "alignment only, never as a model input feature."
            ),
        )
        external_layout.addWidget(self.external_observation_combo)
        self.external_mapping_tree = QTreeWidget()
        self.external_mapping_tree.setColumnCount(2)
        self.external_mapping_tree.setHeaderLabels(
            (
                tr("Development column") + f" {HELP_GLYPH}",
                tr("External column") + f" {HELP_GLYPH}",
            )
        )
        self._help_control(
            self.external_mapping_tree,
            tr(
                "The left column lists what the model was built on; pick the matching "
                "external column on the right. Every input and target must be mapped "
                "explicitly."
            ),
        )
        self.external_mapping_tree.setMinimumHeight(150)
        self.external_mapping_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        external_layout.addWidget(self.external_mapping_tree)
        external_note = QLabel(
            tr(
                "Every model feature and target must be mapped explicitly. The external "
                "target is not read until development CV and Confirmation have finished "
                "and the winning configuration is frozen."
            )
        )
        external_note.setWordWrap(True)
        external_layout.addWidget(external_note)
        layout.addWidget(external, 3, 0, 1, 2)

        output = QGroupBox(tr("Output") + f" {HELP_GLYPH}")
        output.setToolTip(
            tr(
                "The parent folder for every result, figure, Markdown file, PDF and final "
                "model bundle."
            )
        )
        output_form = QFormLayout(output)
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_ROOT))
        self._help_control(
            self.output_edit,
            tr(
                "The parent report folder. Each run creates its own timestamped subfolder "
                "inside it, so earlier results are never overwritten."
            ),
        )
        choose = QPushButton(tr("Browse") + f" {HELP_GLYPH}")
        self._help_control(
            choose, tr("Opens a folder picker and fills in the Master report folder.")
        )
        choose.clicked.connect(self.choose_output_folder)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(choose)
        output_form.addRow(
            self._help_label(
                tr("Master report folder"),
                tr(
                    "The parent directory for every Benchmark Run. Each run creates a "
                    "run_<date>_<time>_<microseconds> subfolder, and never moves or "
                    "overwrites earlier results."
                ),
            ),
            out_row,
        )
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText(
            tr("e.g. shrimp qPCR first pass (blank uses the folder name)")
        )
        self._add_help_row(
            output_form,
            tr("Run name"),
            self.run_name_edit,
            tr(
                "This name is written into every retained model's metadata, and is what "
                "tells training batches with different purposes apart on the Model library "
                "tab."
            ),
        )
        keep_note = QLabel(
            tr(
                "Only the top {limit} models per Target Task are kept. A model whose "
                "Reported R² is not above 0 is never saved, including the first-ranked one.",
                limit=MODEL_KEEP_LIMIT,
            )
        )
        keep_note.setWordWrap(True)
        output_form.addRow(keep_note)
        layout.addWidget(output, 4, 0, 1, 2)
        self._update_augmentation_controls()
        self._update_strategy_controls()
        return tab

    def _apply_run_preset(self) -> None:
        preset = self.run_preset_combo.currentData()
        values = {
            "economy": (250, 20, 1, 0.20, 1.0),
            "balanced": (500, 30, 3, 0.20, 1.0),
            "rigorous": (2000, 60, 5, 0.20, 1.25),
        }.get(preset)
        if values:
            epochs, patience, repetitions, validation, multiplier = values
            self.epochs_spin.setValue(epochs); self.patience_spin.setValue(patience)
            self.repetitions_spin.setValue(repetitions); self.validation_spin.setValue(validation)
            self.model_multiplier_spin.setValue(multiplier)

    def _update_augmentation_controls(self, *_args: object) -> None:
        """Expose policy precedence without destroying the user's method selections."""
        policy = self.augmentation_policy_combo.currentData()
        uses_augmentation = policy != "original_only"
        for control in self.augmentation_detail_controls:
            control.setEnabled(uses_augmentation)
        descriptions = {
            "original_only": tr(
                "Right now only Original runs: every ticked method and its parameters are "
                "ignored. Your ticks are kept, so switching policy resumes them."
            ),
            "compare_separately": tr(
                "Right now this builds the Original baseline plus one configuration per "
                "ticked method. The methods are never mixed."
            ),
            "combined": tr(
                "Right now this builds the Original baseline plus one configuration "
                "combining every ticked method. The report can judge the combination as a "
                "whole, but cannot attribute the effect to one method."
            ),
        }
        self.augmentation_policy_description_label.setText(descriptions.get(policy, ""))
        if hasattr(self, "confirmation_controls"):
            self._update_strategy_controls()

    def _update_strategy_controls(self, *_args: object) -> None:
        """Show which settings participate in the selected evidence workflow."""
        strategy = self.strategy_combo.currentData()
        cv_strategy = strategy in {
            EvaluationStrategy.NESTED_CV,
            EvaluationStrategy.EXTERNAL_VALIDATION,
        }
        holdout = strategy == EvaluationStrategy.HOLDOUT
        final_only = strategy == EvaluationStrategy.TRAIN_FINAL_ONLY
        external = strategy == EvaluationStrategy.EXTERNAL_VALIDATION

        self.test_spin.setEnabled(holdout)
        self.folds_spin.setEnabled(cv_strategy)
        self.run_confirmation_check.setEnabled(cv_strategy)
        confirmation_enabled = cv_strategy and self.run_confirmation_check.isChecked()
        for control in self.confirmation_controls:
            control.setEnabled(confirmation_enabled)
        if self.augmentation_policy_combo.currentData() == "original_only":
            self.confirmation_aug_check.setEnabled(False)
        self.refit_check.setEnabled(not final_only)
        self.external_validation_section.setEnabled(external)

        descriptions = {
            EvaluationStrategy.NESTED_CV: tr(
                "This is an evaluation workflow, not a preset. It uses Screening folds, "
                "with optional Confirmation. Test ratio and External mapping are ignored; "
                "every other training, model, loss and augmentation setting applies as "
                "usual."
            ),
            EvaluationStrategy.HOLDOUT: tr(
                "This is an evaluation workflow, not a preset. It uses Test ratio for one "
                "Train/Test split. Screening folds, every Confirmation setting and "
                "External mapping are ignored. At Test ratio = 0% it performs the final "
                "fit only."
            ),
            EvaluationStrategy.EXTERNAL_VALIDATION: tr(
                "This is an evaluation workflow, not a preset. It selects a configuration "
                "on the development data with Screening folds and optional Confirmation, "
                "then evaluates once through External mapping. Test ratio is ignored."
            ),
            EvaluationStrategy.TRAIN_FINAL_ONLY: tr(
                "This is an evaluation workflow, not a preset. It trains the final models "
                "straight from every development row. Test ratio, Screening folds, "
                "Confirmation, External mapping and the refit option are all ignored, and "
                "no generalization metric is produced."
            ),
        }
        self.strategy_description_label.setText(descriptions.get(strategy, ""))

    def _build_run_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        buttons = QHBoxLayout()
        self.start_button = QPushButton(tr("Start"))
        self.start_button.clicked.connect(self.start_training)
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.current_run_label = QLabel(tr("Training log — not started"))
        self.current_run_label.setWordWrap(True)
        layout.addWidget(self.current_run_label)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)
        return tab

    def browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Choose a data file"),
            str(Path.cwd()),
            tr("Data ({patterns})", patterns="*.xlsx *.xls *.csv"),
        )
        if path:
            self.load_dataset(path)

    def browse_external_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Choose the independent External Validation data file"),
            str(Path.cwd()),
            tr("Data ({patterns})", patterns="*.xlsx *.xls *.csv"),
        )
        if path:
            self.load_external_dataset(path)

    def load_external_dataset(self, path: str) -> None:
        try:
            self.external_bundle = FlexibleDatasetLoader().load(path)
            self.external_dataset_label.setText(path)
            self._refresh_external_mapping()
        except Exception as exc:
            QMessageBox.critical(self, tr("Could not load the external data"), str(exc))

    def _external_column_refs(self) -> tuple[ColumnRef, ...]:
        if self.external_bundle is None:
            return ()
        return tuple(
            ColumnRef(sheet_name, column_name)
            for sheet_name, loaded in self.external_bundle.sheets.items()
            for column_name in loaded.profile.column_profiles
        )

    def _refresh_external_mapping(self) -> None:
        previous = {
            development: combo.currentData()
            for development, combo in self.external_mapping_boxes.items()
            if combo.currentData() is not None
        }
        self.external_mapping_tree.clear()
        self.external_mapping_boxes.clear()
        self.external_observation_combo.clear()
        self.external_observation_combo.addItem(tr("Use the external row position"), None)
        candidates = self._external_column_refs()
        for ref in candidates:
            self.external_observation_combo.addItem(ref.canonical_name, ref)
        if self.external_bundle is None:
            return
        records = self.catalog.selected_records or self.catalog.records
        required = sorted(
            {
                canonical
                for record in records
                for canonical in (*record.input_columns, *record.output_columns)
            }
        )
        by_canonical = {ref.canonical_name: ref for ref in candidates}
        by_column: dict[str, list[ColumnRef]] = {}
        for ref in candidates:
            by_column.setdefault(ref.column, []).append(ref)
        for development in required:
            item = QTreeWidgetItem((development, ""))
            self.external_mapping_tree.addTopLevelItem(item)
            combo = QComboBox()
            combo.addItem(tr("— select —"), None)
            for ref in candidates:
                combo.addItem(ref.canonical_name, ref)
            suggested = previous.get(development) or by_canonical.get(development)
            if suggested is None:
                column_name = development.split("::", 1)[-1]
                matches = by_column.get(column_name, [])
                suggested = matches[0] if len(matches) == 1 else None
            if suggested is not None:
                position = combo.findData(suggested)
                combo.setCurrentIndex(max(0, position))
            self.external_mapping_tree.setItemWidget(item, 1, combo)
            self.external_mapping_boxes[development] = combo
        self.external_mapping_tree.resizeColumnToContents(0)

    def _collect_external_mapping(
        self,
        jobs: Sequence[tuple[Any, ExperimentRecord]],
    ) -> tuple[dict[str, ColumnRef], ColumnRef | None]:
        required = {
            ref.canonical_name
            for spec, _record in jobs
            for ref in (*spec.input_columns, *spec.output_columns)
        }
        missing_boxes = required - self.external_mapping_boxes.keys()
        if missing_boxes:
            self._refresh_external_mapping()
        mapping = {
            development: combo.currentData()
            for development, combo in self.external_mapping_boxes.items()
            if development in required and combo.currentData() is not None
        }
        missing = sorted(required - mapping.keys())
        if missing:
            raise ValueError(
                tr("External mapping is incomplete: {names}", names=", ".join(missing))
            )
        return mapping, self.external_observation_combo.currentData()

    def load_dataset(self, path: str) -> None:
        try:
            self.bundle = FlexibleDatasetLoader().load(path)
            self.dataset_path = path
            self.external_bundle = None
            self.external_dataset_label.setText(
                tr("Not loaded; used by the External Validation strategy only")
            )
            self._refresh_external_mapping()
            self.dataset_label.setText(path)
            self.legacy_specs = []
            self.catalog.replace(())
            self.selection_memory.clear()
            self.structure_memory.clear()
            self.table_model.refresh()
            self._populate_roles()
            rows = sum(sheet.profile.cleaned_rows for sheet in self.bundle.sheets.values())
            self.dataset_summary.setText(
                tr(
                    "Loaded {sheets} sheet(s) / table(s), {rows:,} rows in total. Name an "
                    "Observation ID; if one subject has repeated measurements, name a "
                    "Group ID as well to prevent split leakage.",
                    sheets=len(self.bundle.sheets),
                    rows=rows,
                )
            )
        except Exception as exc:
            QMessageBox.critical(self, tr("Could not load the data"), str(exc))

    def _populate_roles(self) -> None:
        assert self.bundle is not None
        roles = role_options()
        self.role_tree.clear()
        self.sheet_role_boxes.clear()
        self.column_role_boxes.clear()
        self.observation_ref = None
        self.group_ref = None
        for sheet_name, loaded in self.bundle.sheets.items():
            profile = loaded.profile
            parent = QTreeWidgetItem((sheet_name, "", "Sheet", str(profile.cleaned_rows), str(profile.total_missing_cells), f"header={profile.header_detected}"))
            self.role_tree.addTopLevelItem(parent)
            sheet_combo = QComboBox()
            for value in ("skip", "input", "combined_input", "output", "input_or_output"):
                sheet_combo.addItem(roles[value], value)
            self.role_tree.setItemWidget(parent, 1, sheet_combo)
            self.sheet_role_boxes[sheet_name] = sheet_combo
            for column_name, column in profile.column_profiles.items():
                child = QTreeWidgetItem((
                    column_name,
                    "",
                    tr("numeric") if column.trainable else tr("ID / text candidate"),
                    str(profile.cleaned_rows),
                    str(column.missing),
                    f"unique={column.unique_values}; numeric={column.numeric_ratio:.1%}",
                ))
                parent.addChild(child)
                combo = QComboBox()
                values = ["skip", "observation_id", "group_id"]
                if column.trainable:
                    values[1:1] = ["input", "combined_input", "output", "input_or_output"]
                for value in values:
                    combo.addItem(roles[value], value)
                combo.currentIndexChanged.connect(lambda _index, s=sheet_name, c=column_name: self._column_role_changed(s, c))
                self.role_tree.setItemWidget(child, 1, combo)
                self.column_role_boxes[(sheet_name, column_name)] = combo
            sheet_combo.currentIndexChanged.connect(lambda _index, s=sheet_name: self._sheet_role_changed(s))
            parent.setExpanded(True)
        self._apply_identity_suggestions()
        self.role_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.role_tree.setColumnWidth(0, 260)
        self.role_tree.setColumnWidth(1, 220)

    def _apply_identity_suggestions(self) -> None:
        """Conservatively protect obvious ID columns from accidental feature use."""

        if self.bundle is None:
            return
        observation_tokens = {"id", "編號", "编号", "樣本id", "样本id", "sampleid", "observationid"}
        group_tokens = {"groupid", "subjectid", "個體id", "个体id", "受試者id", "受试者id"}
        suggested_observation = False
        suggested_group = False
        for (sheet, column), combo in self.column_role_boxes.items():
            normalized = "".join(character.lower() for character in column if character.isalnum())
            if not suggested_observation and normalized in observation_tokens:
                index = combo.findData("observation_id")
                if index >= 0:
                    combo.setCurrentIndex(index)
                    combo.setToolTip(
                        tr(
                            "Suggested as the Observation ID from the column name — please "
                            "confirm. This column never becomes a model feature."
                        )
                    )
                    suggested_observation = True
            elif not suggested_group and normalized in group_tokens:
                index = combo.findData("group_id")
                if index >= 0:
                    combo.setCurrentIndex(index)
                    combo.setToolTip(
                        tr("Suggested as the Group ID from the column name — please confirm.")
                    )
                    suggested_group = True

    def _sheet_role_changed(self, sheet: str) -> None:
        if self.bundle is None:
            return
        inherited = self.sheet_role_boxes[sheet].currentData() != "skip"
        for column in self.bundle.sheets[sheet].profile.column_profiles:
            combo = self.column_role_boxes[(sheet, column)]
            combo.setEnabled(not inherited)
            if inherited:
                combo.setCurrentIndex(0)
        self._invalidate_experiments()

    def _column_role_changed(self, sheet: str, column: str) -> None:
        if self._role_change_guard:
            return
        value = self.column_role_boxes[(sheet, column)].currentData()
        if value in {"observation_id", "group_id"}:
            self._role_change_guard = True
            for (other_sheet, other_column), combo in self.column_role_boxes.items():
                if (other_sheet, other_column) != (sheet, column) and combo.currentData() == value:
                    combo.setCurrentIndex(0)
            self._role_change_guard = False
        self._invalidate_experiments()

    def _invalidate_experiments(self) -> None:
        self.selection_memory.update({record.stable_key: record.selected for record in self.catalog.records})
        self.legacy_specs = []
        self.catalog.replace(())
        self.table_model.refresh()
        self.combo_label.setText(tr("Roles changed — regenerate the combinations"))

    def _to_data_role(self, value: str) -> DataRole:
        return {
            "input": DataRole.INPUT,
            "combined_input": DataRole.COMBINED_INPUT,
            "output": DataRole.OUTPUT,
            "input_or_output": DataRole.FLEX,
        }.get(value, DataRole.SKIP)

    def collect_units(self) -> list[SelectionUnit]:
        assert self.bundle is not None
        units: list[SelectionUnit] = []
        self.observation_ref = None
        self.group_ref = None
        for sheet in self.bundle.sheets:
            sheet_role = self._to_data_role(self.sheet_role_boxes[sheet].currentData())
            if sheet_role != DataRole.SKIP:
                refs = tuple(ColumnRef(sheet, column) for column in self.bundle.trainable_columns(sheet))
                if refs:
                    units.append(SelectionUnit(f"sheet::{sheet}", sheet, sheet_role, refs, "sheet"))
                continue
            combined: list[ColumnRef] = []
            for column in self.bundle.sheets[sheet].profile.column_profiles:
                value = self.column_role_boxes[(sheet, column)].currentData()
                ref = ColumnRef(sheet, column)
                if value == "observation_id":
                    self.observation_ref = ref
                elif value == "group_id":
                    self.group_ref = ref
                else:
                    role = self._to_data_role(value)
                    if role == DataRole.COMBINED_INPUT:
                        combined.append(ref)
                    elif role != DataRole.SKIP:
                        units.append(SelectionUnit(f"column::{sheet}::{column}", f"{sheet}.{column}", role, (ref,), "column"))
            if combined:
                units.append(SelectionUnit(f"combined::{sheet}", f"{sheet} Combined Input", DataRole.COMBINED_INPUT, tuple(combined), "sheet"))
        return units

    def generate_preview(self) -> None:
        if self.bundle is None:
            QMessageBox.warning(
                self, tr("No data loaded"), tr("Load a CSV or Excel file first.")
            )
            return
        try:
            specs = generate_experiment_specs(self.collect_units(), self.bundle, max_combinations=0, min_valid_rows=8)
            records = [ExperimentRecord.from_legacy(spec) for spec in specs]
            self.catalog.replace(records)
            for key, selected in self.selection_memory.items():
                if any(record.stable_key == key for record in self.catalog.records):
                    self.catalog.set_selected(key, selected)
            for key, (structure, confirmed) in self.structure_memory.items():
                if any(record.stable_key == key for record in self.catalog.records):
                    self.catalog.set_feature_structure(
                        key,
                        structure,
                        confirmed=confirmed,
                    )
            self.legacy_specs = specs
            self.table_model.refresh()
            self._update_selection_label()
            self._refresh_external_mapping()
            if not specs:
                QMessageBox.warning(
                self,
                tr("No valid combination"),
                tr("At least one input and one output are required."),
            )
        except Exception as exc:
            QMessageBox.critical(self, tr("Could not generate combinations"), str(exc))

    def _set_selection(self, mode: str) -> None:
        self.table_model.set_all(mode)
        self._update_selection_label()

    def _confirm_selected_structures(self) -> None:
        for record in self.catalog.selected_records:
            self.catalog.set_feature_structure(
                record.stable_key,
                record.feature_structure,
                confirmed=True,
            )
        self.table_model.refresh()
        self._update_selection_label()

    def _update_selection_label(self, *_args: Any) -> None:
        self.selection_memory.update({record.stable_key: record.selected for record in self.catalog.records})
        self.structure_memory.update(
            {
                record.stable_key: (record.feature_structure, record.structure_confirmed)
                for record in self.catalog.records
            }
        )
        confirmed = sum(record.structure_confirmed for record in self.catalog.selected_records)
        self.combo_label.setText(
            tr(
                "{selected:,} of {total:,} selected; {confirmed:,} structure(s) confirmed",
                selected=len(self.catalog.selected_records),
                total=len(self.catalog.records),
                confirmed=confirmed,
            )
        )

    def choose_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, tr("Choose the output folder"), self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)

    @staticmethod
    def _create_run_output_folder(
        base_folder: str | Path,
        *,
        now: datetime | None = None,
    ) -> Path:
        """Atomically create a timestamped child for one Benchmark Run."""
        base = Path(base_folder).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
        stem = f"run_{timestamp}"
        suffix = 0
        while True:
            name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
            candidate = base / name
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                suffix += 1
                continue
            return candidate

    def collect_config(self) -> EvaluationConfig:
        strategy = self.strategy_combo.currentData()
        cv_strategy = strategy in {
            EvaluationStrategy.NESTED_CV,
            EvaluationStrategy.EXTERNAL_VALIDATION,
        }
        augmentation_policy = self.augmentation_policy_combo.currentData()
        selected_augmentations = (
            ()
            if augmentation_policy == "original_only"
            else tuple(
                key for key, check in self.augmentation_checks.items() if check.isChecked()
            )
        )
        run_confirmation = cv_strategy and self.run_confirmation_check.isChecked()
        return EvaluationConfig(
            strategy=strategy,
            test_size=self.test_spin.value() / 100.0,
            screening_folds=self.folds_spin.value(),
            random_seed=self.seed_spin.value(),
            refit_on_all_data=(
                True
                if strategy == EvaluationStrategy.TRAIN_FINAL_ONLY
                else self.refit_check.isChecked()
            ),
            confirmation_folds=self.confirmation_outer_folds_spin.value(),
            confirmation_repeats=self.confirmation_outer_repeats_spin.value(),
            confirmation_inner_folds=self.confirmation_inner_folds_spin.value(),
            confirmation_tuning_budget=self.tuning_budget_spin.value(),
            finalists_per_task=self.finalists_spin.value(),
            run_confirmation=run_confirmation,
            confirmation_augmentation_ablation=(
                run_confirmation
                and augmentation_policy != "original_only"
                and self.confirmation_aug_check.isChecked()
            ),
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            model_multiplier=self.model_multiplier_spin.value(),
            repetitions=self.repetitions_spin.value(),
            validation_ratio=self.validation_spin.value(),
            selected_scalers=tuple(key for key, check in self.scaler_checks.items() if check.isChecked()),
            selected_losses=tuple(key for key, check in self.loss_checks.items() if check.isChecked()),
            augmentation_policy=augmentation_policy,
            selected_augmentations=selected_augmentations,
            augmentation_repeats=self.augmentation_repeats_spin.value(),
            augmentation_ratio=self.augmentation_ratio_spin.value(),
            augmentation_include_original=self.augmentation_include_original_check.isChecked(),
            c_mixup_alpha=self.c_mixup_alpha_spin.value(),
            noise_fraction=self.noise_fraction_spin.value(),
            feature_mask_probability=self.feature_mask_probability_spin.value(),
            foma_alpha=self.foma_alpha_spin.value(),
            foma_k=self.foma_k_spin.value(),
        )

    def start_training(self) -> None:
        if self.bundle is None:
            QMessageBox.warning(
                self,
                tr("No data loaded"),
                tr("Load the data and assign column roles first."),
            )
            return
        if not self.legacy_specs:
            self.generate_preview()
        selected = {record.stable_key: record for record in self.catalog.selected_records}
        jobs = []
        for spec in self.legacy_specs:
            record = ExperimentRecord.from_legacy(spec)
            if record.stable_key in selected:
                jobs.append((spec, selected[record.stable_key]))
        models = [model_id for model_id, check in self.model_checks.items() if check.isEnabled() and check.isChecked()]
        if not jobs or not models:
            QMessageBox.warning(
                self,
                tr("Nothing to run"),
                tr("Tick at least one data combination and one available model."),
            )
            return
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.warning(self, tr("Incomplete settings"), str(exc))
            return
        external_mapping: dict[str, ColumnRef] = {}
        external_observation_ref: ColumnRef | None = None
        if config.strategy == EvaluationStrategy.EXTERNAL_VALIDATION:
            if self.external_bundle is None:
                QMessageBox.warning(
                    self,
                    tr("External data required"),
                    tr(
                        "Load the independent External Validation data and finish the "
                        "column mapping."
                    ),
                )
                return
            try:
                external_mapping, external_observation_ref = self._collect_external_mapping(jobs)
            except Exception as exc:
                QMessageBox.warning(self, tr("External mapping is incomplete"), str(exc))
                return
        output_path = self._create_run_output_folder(self.output_edit.text())
        self.current_run_output_folder = output_path
        output = str(output_path)
        self.training_log_path = output_path / "training.log"
        self.training_log_path.write_text("", encoding="utf-8")
        self.log_box.clear()
        self.on_worker_log(f"Run output folder: {output_path}")
        self.progress.setValue(0)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.tabs.setCurrentIndex(2)
        self.worker = BenchmarkWorker(
            self.bundle,
            jobs,
            self.observation_ref,
            self.group_ref,
            models,
            config,
            output,
            self.external_bundle,
            external_mapping,
            external_observation_ref,
            run_label=self.run_name_edit.text().strip() or output_path.name,
        )
        self.worker.log_message.connect(self.on_worker_log)
        self.worker.progress_changed.connect(self.on_progress)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.start()

    def cancel_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.log_box.appendPlainText(
                tr("Cancellation requested; the run stops after the current fold.")
            )

    def on_progress(self, current: int, total: int, label: str) -> None:
        self.progress.setValue(round(100 * current / max(1, total)))
        self.current_run_label.setText(f"Training log — {current}/{total}: {label}")

    def on_worker_log(self, message: str) -> None:
        self.log_box.appendPlainText(message)
        if self.training_log_path is not None:
            with self.training_log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def on_completed(self, artifacts: ReportArtifacts) -> None:
        self._finish_worker()
        self.progress.setValue(100)
        self.run_completed.emit(str(self.current_run_output_folder or ""))
        if artifacts.pdf_path is not None:
            self.current_run_label.setText(
                tr("Training log — completed — {path}", path=str(artifacts.pdf_path))
            )
            QMessageBox.information(
                self,
                tr("Finished"),
                tr("Master report written to:\n{path}", path=str(artifacts.pdf_path)),
            )
        else:
            error_path = getattr(artifacts, "rendering_error_path", None)
            self.current_run_label.setText(
                tr("Training log — completed, but the PDF could not be rendered")
            )
            QMessageBox.warning(
                self,
                tr("Training finished, but the PDF failed"),
                tr(
                    "The CSV, Markdown, PNG, model and audit files are all intact.\n"
                    "Error log: {path}",
                    path=str(error_path),
                ),
            )

    def on_failed(self, details: str) -> None:
        self._finish_worker()
        self.log_box.appendPlainText(details)
        QMessageBox.critical(self, tr("Run failed"), details[-3000:])

    def on_cancelled(self, message: str) -> None:
        self._finish_worker()
        self.current_run_label.setText(message)

    def _finish_worker(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def retranslate(self) -> None:
        """Rebuild the workspace in the new language.

        The configuration tab builds several dozen labels and tooltips inline,
        so rebuilding is both simpler and less error-prone than tracking every
        widget. A loaded dataset is reloaded from its path afterwards; a run in
        progress blocks the switch entirely, which the host window checks first.
        """

        dataset_path = self.dataset_path
        output_folder = self.output_edit.text()
        run_name = self.run_name_edit.text()

        self.bundle = None
        self.external_bundle = None
        self.legacy_specs = []
        self.catalog.replace(())
        self.sheet_role_boxes.clear()
        self.column_role_boxes.clear()
        self.model_checks.clear()
        self.scaler_checks.clear()
        self.loss_checks.clear()
        self.augmentation_checks.clear()
        self.external_mapping_boxes.clear()
        self.observation_ref = None
        self.group_ref = None
        self.setting_help_labels = []
        self.setting_help_controls = []

        previous = self.layout()
        while previous.count():
            item = previous.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        # A widget may own only one layout, and deleteLater() would not detach
        # the old one in time, so hand it to a throwaway parent instead.
        QWidget().setLayout(previous)

        self._build_ui()
        self.output_edit.setText(output_folder)
        self.run_name_edit.setText(run_name)
        if dataset_path and Path(dataset_path).exists():
            self.load_dataset(dataset_path)

    def has_running_worker(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def can_close(self) -> bool:
        """Refuse to close while a Benchmark Run still owns a worker thread."""

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                tr("A run is in progress"),
                tr("Cancel it first and wait for the current fold to finish."),
            )
            return False
        return True


class RegressionV4MainWindow(QMainWindow):
    """Standalone training-only window, kept for debugging the engine alone."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Small-Data Regression Benchmark V4 — {EXECUTION_MODE_LABEL}")
        self.resize(1540, 940)
        self.page = TrainingPage()
        self.setCentralWidget(self.page)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.page.can_close():
            event.accept()
        else:
            event.ignore()


def launch(argv: Sequence[str] | None = None) -> int:
    import i18n

    i18n.initialize()
    app = QApplication(list(argv or []))
    window = RegressionV4MainWindow()
    window.show()
    return app.exec()
