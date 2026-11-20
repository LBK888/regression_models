"""PyQt6 user interface for the first executable V4 benchmark slice."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import replace

import numpy as np

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
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
    QRadioButton,
    QScrollArea,
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
    LOSS_CONFIGURABLE_MODEL_IDS,
    NEURAL_MODEL_IDS,
    adaptive_capacity_scale,
    model_incompatibility_reason,
    model_registry,
)
from .data_audit import audit_columns
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

#: How the model check boxes are grouped on the settings page. Titles are
#: translated when the page is built; a model missing here lands in "Other".
MODEL_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Baselines", ("mean", "ridge", "pls", "svr_rbf")),
    ("Trees and boosting", ("random_forest", "extra_trees", "catboost", "xgboost", "lightgbm")),
    ("Pretrained for small tables", ("tabpfn",)),
    (
        "Neural networks",
        (
            "deep_mlp",
            "wide_mlp",
            "residual_mlp",
            "multi_branch_mlp",
            "bottleneck_mlp",
            "mlp_embeddings",
            "ft_transformer",
            "modern_nca",
            "tabm",
            "realmlp",
        ),
    ),
    ("Ordered or grouped inputs only", ("cnn1d", "resnet1d", "grouped_fusion")),
)

#: Column names (lower-case, letters and digits only) that usually mark rows
#: measured together. A repeated column with one of these names is suggested as
#: the Group ID so cross-validation never splits one day across train and test.
REPEATED_MEASURE_TOKENS = frozenset(
    {
        "date", "day", "sampledate", "samplingdate", "measurementdate",
        "日期", "採樣日期", "采样日期", "取樣日期", "量測日期", "測量日期",
        "batch", "batchid", "批次", "tank", "tankid", "pond", "池號", "缸號",
    }
)

#: The "Recommended" button: strong, cheap models for a few hundred rows.
RECOMMENDED_MODEL_IDS = frozenset(
    {"mean", "ridge", "pls", "svr_rbf", "random_forest", "extra_trees", "catboost", "tabpfn"}
)

#: Ticked when the page opens.
DEFAULT_MODEL_IDS = frozenset({"mean", "ridge", "pls", "svr_rbf", "extra_trees", "tabpfn"})


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
        "tabpfn": tr(
            "TabPFN v2: a network pre-trained on millions of synthetic tables that "
            "predicts a new table without per-dataset training. Usually one of the "
            "strongest models below a few thousand rows. Uses the v2 weights, which need "
            "no account; they download once (about 45 MB)."
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
        layout.setContentsMargins(10, 2, 10, 2)
        label = QLabel(tr("Drop a CSV / Excel file here"))
        label.setObjectName("hint")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setMinimumHeight(34)

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
        self.toggle_button.setObjectName("sectionToggle")
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
                if not identity.is_grouped:
                    self.log_message.emit(
                        tr(
                            "  Warning: no Group ID. If several rows come from the same day, "
                            "tank or subject, this score is optimistic."
                        )
                    )
                if (
                    self.config.strategy == EvaluationStrategy.HOLDOUT
                    and self.config.holdout_chronological
                ):
                    self.log_message.emit(
                        tr("  Holdout uses the most recent rows (data order) as the test set.")
                    )
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

    # ------------------------------------------------------------ data tab
    def _build_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        row = QHBoxLayout()
        drop = DatasetDropFrame()
        drop.file_dropped.connect(self.load_dataset)
        browse = QPushButton(tr("Choose a data file"))
        browse.clicked.connect(self.browse_dataset)
        self.dataset_label = QLabel(tr("No data loaded"))
        self.dataset_label.setObjectName("hint")
        self.dataset_label.setMinimumWidth(80)
        preview = QPushButton(tr("Generate / refresh combinations"))
        preview.clicked.connect(self.generate_preview)
        row.addWidget(drop)
        row.addWidget(browse)
        row.addWidget(self.dataset_label, 1)
        row.addWidget(preview)
        layout.addLayout(row)
        self.dataset_summary = QLabel(
            tr("Observation ID and Group ID never become model features.")
        )
        self.dataset_summary.setWordWrap(True)
        layout.addWidget(self.dataset_summary)
        self.data_group_status_label = QLabel()
        self.data_group_status_label.setWordWrap(True)
        layout.addWidget(self.data_group_status_label)

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
        right_layout.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.combo_label = QLabel(tr("No combinations generated yet"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            tr("Filter by ID, Inputs, Outputs or Stable Key…")
        )
        all_button = QPushButton(tr("Select all"))
        none_button = QPushButton(tr("Select none"))
        invert_button = QPushButton(tr("Invert"))
        confirm_structures_button = QPushButton(tr("Confirm structure"))
        confirm_structures_button.setToolTip(
            tr("Confirm the suggested structure for the selected combinations")
        )
        for button in (all_button, none_button, invert_button, confirm_structures_button):
            button.setObjectName("secondaryBtn")
        tools.addWidget(self.filter_edit, 1)
        tools.addWidget(all_button)
        tools.addWidget(none_button)
        tools.addWidget(invert_button)
        tools.addWidget(confirm_structures_button)
        right_layout.addWidget(self.combo_label)
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
        splitter.setSizes((660, 620))
        layout.addWidget(splitter, 1)
        return tab

    # --------------------------------------------------------- form helpers
    @staticmethod
    def _hint(text: str = "", *, object_name: str = "hint") -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _compact_form() -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return form

    @staticmethod
    def _plain_container() -> QWidget:
        widget = QWidget()
        widget.setObjectName("plain")
        return widget

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int, *, suffix: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(120)
        return spin

    @staticmethod
    def _float_spin(
        minimum: float,
        maximum: float,
        value: float,
        *,
        step: float,
        decimals: int = 2,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(120)
        return spin

    def _form_row(self, form: QFormLayout, text: str, control: QWidget, explanation: str) -> QLabel:
        """Add a labelled row; the explanation is the tooltip on both halves."""

        label = QLabel(text)
        label.setToolTip(explanation)
        control.setToolTip(explanation)
        self.setting_help_labels.append(label)
        self.setting_help_controls.append(control)
        form.addRow(label, control)
        return label

    def _tip(self, control: QWidget, explanation: str) -> QWidget:
        control.setToolTip(explanation)
        self.setting_help_controls.append(control)
        return control

    # ------------------------------------------------------ settings page
    def _build_config_tab(self) -> QWidget:
        """Settings in decision order; only what the current choices use is shown.

        Every value survives being hidden: switching strategy, models or the
        augmentation mode only changes visibility, never the numbers typed in.
        """

        self._settings_ready = False
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("settingsPage")
        columns = QHBoxLayout(content)
        columns.setContentsMargins(8, 4, 8, 8)
        columns.setSpacing(10)
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.setSpacing(8)
        right.setSpacing(8)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        scroll.setWidget(content)

        left.addWidget(self._build_evaluation_box())
        left.addWidget(self._build_neural_box())
        left.addWidget(self._build_preparation_box())
        left.addStretch(1)
        right.addWidget(self._build_models_box())
        right.addWidget(self._build_augmentation_section())
        right.addWidget(self._build_output_box())
        self.run_estimate_label = self._hint()
        right.addWidget(self.run_estimate_label)
        right.addStretch(1)

        self._settings_ready = True
        self._refresh_settings_visibility()
        return scroll

    def _build_evaluation_box(self) -> QWidget:
        box = QGroupBox(tr("1 · How to measure accuracy"))
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self.group_status_label = QLabel()
        self.group_status_label.setWordWrap(True)
        layout.addWidget(self.group_status_label)

        self.strategy_group = QButtonGroup(self)
        self.strategy_buttons: dict[EvaluationStrategy, QRadioButton] = {}
        self.strategy_descriptions = {
            EvaluationStrategy.NESTED_CV: tr(
                "Every row is predicted once by a model that never saw it. The most "
                "reliable score for small data sets."
            ),
            EvaluationStrategy.HOLDOUT: tr(
                "Sets part of the data aside once and scores the models on it. Quick, "
                "but on small data the score depends a lot on which rows were picked."
            ),
            EvaluationStrategy.EXTERNAL_VALIDATION: tr(
                "Picks the best model by cross-validation on this data, then scores it "
                "once on a second, independent data file."
            ),
            EvaluationStrategy.TRAIN_FINAL_ONLY: tr(
                "Fits the models on every row and saves them without any score, so "
                "nothing tells you whether they actually work."
            ),
        }
        titles = (
            (EvaluationStrategy.NESTED_CV, tr("Cross-validation (recommended)")),
            (EvaluationStrategy.HOLDOUT, tr("One train / test split")),
            (EvaluationStrategy.EXTERNAL_VALIDATION, tr("Score on a separate data file")),
            (EvaluationStrategy.TRAIN_FINAL_ONLY, tr("Train only, no score")),
        )
        radios = QGridLayout()
        radios.setHorizontalSpacing(16)
        radios.setVerticalSpacing(2)
        for index, (strategy, title) in enumerate(titles):
            button = QRadioButton(title)
            button.setToolTip(self.strategy_descriptions[strategy])
            self.strategy_group.addButton(button, index)
            self.strategy_buttons[strategy] = button
            radios.addWidget(button, index // 2, index % 2)
        self.strategy_buttons[EvaluationStrategy.NESTED_CV].setChecked(True)
        layout.addLayout(radios)
        self.strategy_description_label = self._hint()
        layout.addWidget(self.strategy_description_label)

        form = self._compact_form()
        self.strategy_form = form
        self.folds_spin = self._int_spin(2, 20, 5)
        self.test_spin = self._float_spin(0.0, 50.0, 20.0, step=5.0, decimals=0, suffix=" %")
        self.chronological_check = QCheckBox(tr("Use the most recent rows as the test set"))
        self.seed_spin = self._int_spin(0, 2_147_483_647, 42)
        self.refit_check = QCheckBox(tr("Retrain the kept models on all rows before saving"))
        self.refit_check.setChecked(True)
        self.run_confirmation_check = QCheckBox(
            tr("Double-check the best candidates with nested cross-validation (slow)")
        )
        self._form_row(
            form, tr("Folds (K)"), self.folds_spin,
            tr(
                "The data is cut into K parts; each part is predicted once by a model "
                "trained on the other K−1 parts. 5 is a good default. With a Group ID, K "
                "cannot exceed the number of groups."
            ),
        )
        self._form_row(
            form, tr("Test share"), self.test_spin,
            tr(
                "The share of rows (or of groups) set aside for scoring. At 0 % every row "
                "is used for training and no score is produced."
            ),
        )
        self._tip(
            self.chronological_check,
            tr(
                "Instead of random test rows, keep the last rows of the file (or the last "
                "groups) for testing: the model is trained on the past and scored on the "
                "future. This shows how it copes when conditions drift over time."
            ),
        )
        form.addRow(self.chronological_check)
        self._form_row(
            form, tr("Random seed"), self.seed_spin,
            tr(
                "Fixes the random choices (splits, initial weights, synthetic rows). The "
                "same data, settings and seed reproduce the same result."
            ),
        )
        self._tip(
            self.refit_check,
            tr(
                "The score comes from the evaluation. Afterwards the kept models are "
                "trained once more on every row, so the saved models use all the data."
            ),
        )
        form.addRow(self.refit_check)
        self._tip(
            self.run_confirmation_check,
            tr(
                "After the first round, the best few candidates per target are re-scored "
                "with repeated nested cross-validation, tuning their settings without ever "
                "touching the test folds. More trustworthy, many times slower."
            ),
        )
        form.addRow(self.run_confirmation_check)
        layout.addLayout(form)

        self.confirmation_panel = self._plain_container()
        confirmation_form = self._compact_form()
        confirmation_form.setContentsMargins(24, 0, 0, 0)
        self.confirmation_panel.setLayout(confirmation_form)
        self.confirmation_form = confirmation_form
        self.finalists_spin = self._int_spin(1, 20, 5)
        self.confirmation_outer_folds_spin = self._int_spin(2, 10, 5)
        self.confirmation_outer_repeats_spin = self._int_spin(1, 20, 3)
        self.confirmation_inner_folds_spin = self._int_spin(2, 10, 3)
        self.tuning_budget_spin = self._int_spin(1, 100, 8)
        self.confirmation_aug_check = QCheckBox(
            tr("Also re-check whether augmentation really helps")
        )
        self.confirmation_aug_check.setChecked(True)
        self._form_row(
            confirmation_form, tr("Candidates per target"), self.finalists_spin,
            tr("How many of the best first-round configurations per target are re-checked."),
        )
        self._form_row(
            confirmation_form, tr("Outer folds"), self.confirmation_outer_folds_spin,
            tr("Folds of the outer cross-validation that produces the confirmed score."),
        )
        self._form_row(
            confirmation_form, tr("Repeats"), self.confirmation_outer_repeats_spin,
            tr(
                "Repeats the outer cross-validation with new random splits, which shows "
                "how much the score varies. Multiplies the run time."
            ),
        )
        self._form_row(
            confirmation_form, tr("Inner folds"), self.confirmation_inner_folds_spin,
            tr(
                "Folds used inside each outer training part to choose settings, so the "
                "outer test fold never influences the choice."
            ),
        )
        self._form_row(
            confirmation_form, tr("Settings tried"), self.tuning_budget_spin,
            tr("How many setting combinations each candidate may try per outer fold."),
        )
        self._tip(
            self.confirmation_aug_check,
            tr(
                "Re-scores the candidate with and without the selected augmentation on "
                "identical folds, so you can see whether the synthetic rows really help."
            ),
        )
        confirmation_form.addRow(self.confirmation_aug_check)
        layout.addWidget(self.confirmation_panel)

        layout.addWidget(self._build_external_panel())
        self.strategy_group.idToggled.connect(
            lambda _id, checked: checked and self._refresh_settings_visibility()
        )
        self.run_confirmation_check.toggled.connect(self._refresh_settings_visibility)
        return box

    def _build_external_panel(self) -> QWidget:
        panel = self._plain_container()
        self.external_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        top = QHBoxLayout()
        external_button = QPushButton(tr("Load the separate data file"))
        external_button.setObjectName("secondaryBtn")
        self._tip(
            external_button,
            tr(
                "A file that took no part in choosing the model. It is read only for the "
                "single final score."
            ),
        )
        external_button.clicked.connect(self.browse_external_dataset)
        self.external_dataset_label = self._hint(tr("No separate file loaded yet"))
        top.addWidget(external_button)
        top.addWidget(self.external_dataset_label, 1)
        layout.addLayout(top)
        form = self._compact_form()
        self.external_observation_combo = QComboBox()
        self.external_observation_combo.addItem(tr("Use the external row position"), None)
        self._form_row(
            form, tr("Row ID in that file (optional)"), self.external_observation_combo,
            tr(
                "A column that identifies each row of the separate file. It is only used "
                "to line rows up in the report, never as a model input."
            ),
        )
        layout.addLayout(form)
        self.external_mapping_tree = QTreeWidget()
        self.external_mapping_tree.setColumnCount(2)
        self.external_mapping_tree.setHeaderLabels(
            (tr("Column in this data"), tr("Matching column in the separate file"))
        )
        self._tip(
            self.external_mapping_tree,
            tr(
                "Pick, for every input and target, the matching column of the separate "
                "file. Every one must be mapped."
            ),
        )
        self.external_mapping_tree.setMinimumHeight(130)
        self.external_mapping_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        layout.addWidget(self.external_mapping_tree)
        layout.addWidget(
            self._hint(
                tr(
                    "The separate file's targets are read only after the best model has "
                    "been chosen and frozen."
                )
            )
        )
        return panel

    def _build_models_box(self) -> QWidget:
        box = QGroupBox(tr("2 · Models to compare"))
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        quick = QHBoxLayout()
        for text, mode in (
            (tr("Recommended"), "recommended"),
            (tr("All available"), "all"),
            (tr("None"), "none"),
        ):
            button = QPushButton(text)
            button.setObjectName("secondaryBtn")
            button.clicked.connect(lambda _checked=False, value=mode: self._quick_select_models(value))
            quick.addWidget(button)
        quick.addStretch(1)
        self.model_count_label = self._hint()
        quick.addWidget(self.model_count_label)
        layout.addLayout(quick)

        helps = model_help()
        registry = {capability.model_id: capability for capability in model_registry()}
        listed = {model_id for _title, ids in MODEL_SECTIONS for model_id in ids}
        sections = [*MODEL_SECTIONS]
        leftovers = tuple(model_id for model_id in registry if model_id not in listed)
        if leftovers:
            sections.append(("Other", leftovers))
        for title, ids in sections:
            capabilities = [registry[model_id] for model_id in ids if model_id in registry]
            if not capabilities:
                continue
            header = QLabel(tr(title))
            header.setObjectName("sectionTitle")
            layout.addWidget(header)
            grid = QGridLayout()
            grid.setContentsMargins(8, 0, 0, 2)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(2)
            for index, capability in enumerate(capabilities):
                check = QCheckBox(capability.display_name)
                check.setEnabled(capability.enabled)
                check.setChecked(capability.enabled and capability.model_id in DEFAULT_MODEL_IDS)
                explanation = helps.get(
                    capability.model_id,
                    tr("Adds this model to every compatible combination."),
                )
                if not capability.enabled:
                    check.setText(tr("{name} (not installed)", name=capability.display_name))
                    explanation += " " + tr(
                        "Currently unavailable: {reason}", reason=capability.reason
                    )
                self._tip(check, explanation)
                check.toggled.connect(self._refresh_settings_visibility)
                self.model_checks[capability.model_id] = check
                grid.addWidget(check, index // 3, index % 3)
            layout.addLayout(grid)
        return box

    def _build_neural_box(self) -> QWidget:
        box = QGroupBox(tr("3 · Neural-network training"))
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self.neural_status_label = self._hint()
        layout.addWidget(self.neural_status_label)
        self.neural_settings = self._plain_container()
        form = self._compact_form()
        self.neural_form = form
        self.neural_settings.setLayout(form)

        self.run_preset_combo = QComboBox()
        for label, value in (
            (tr("Quick"), "economy"),
            (tr("Balanced"), "balanced"),
            (tr("Thorough"), "rigorous"),
            (tr("Custom"), "custom"),
        ):
            self.run_preset_combo.addItem(label, value)
        self.run_preset_combo.setCurrentIndex(1)
        self.epochs_spin = self._int_spin(1, 1_000_000, 500)
        self.patience_spin = self._int_spin(1, 100_000, 30, suffix=tr(" epochs"))
        self.batch_spin = self._int_spin(2, 100_000, 32)
        self.learning_rate_spin = self._float_spin(1e-7, 1.0, 0.001, step=0.0005, decimals=5)
        self.validation_spin = self._float_spin(5.0, 50.0, 20.0, step=5.0, decimals=0, suffix=" %")
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItem(tr("Auto — grows with the training rows"), "auto")
        self.model_size_combo.addItem(tr("Fixed"), "fixed")
        self.model_multiplier_spin = self._float_spin(0.10, 20.0, 1.0, step=0.25, suffix=" ×")
        self.model_size_hint = self._hint()
        loss_row = self._plain_container()
        loss_layout = QHBoxLayout(loss_row)
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_texts = loss_help()
        for loss_id, label, checked in (
            ("mse", "MSE", True),
            ("huber", "Huber", True),
            ("mae", "MAE", False),
            ("log_cosh", "Log-cosh", False),
        ):
            check = QCheckBox(label)
            check.setChecked(checked)
            self._tip(check, loss_texts[loss_id])
            check.toggled.connect(self._refresh_settings_visibility)
            self.loss_checks[loss_id] = check
            loss_layout.addWidget(check)
        loss_layout.addStretch(1)

        self._form_row(
            form, tr("Preset"), self.run_preset_combo,
            tr(
                "Fills in Max epochs and Patience. Quick 250/20, Balanced 500/30, "
                "Thorough 2000/60. Editing either value switches to Custom."
            ),
        )
        self._form_row(
            form, tr("Max epochs"), self.epochs_spin,
            tr(
                "Upper limit on passes over the training rows. Training usually stops "
                "earlier through early stopping."
            ),
        )
        self._form_row(
            form, tr("Patience"), self.patience_spin,
            tr(
                "Stop when the early-stopping score has not improved for this many epochs, "
                "and keep the best epoch."
            ),
        )
        self._form_row(
            form, tr("Batch size"), self.batch_spin,
            tr("Rows per weight update. Smaller is noisier, larger is faster."),
        )
        self._form_row(
            form, tr("Learning rate"), self.learning_rate_spin,
            tr("Step size of each weight update. Too large is unstable, too small is slow."),
        )
        self._form_row(
            form, tr("Early-stopping share"), self.validation_spin,
            tr(
                "Share of each training part watched for early stopping. It is taken from "
                "the training rows only; test rows are never used."
            ),
        )
        self._form_row(
            form, tr("Model size"), self.model_size_combo,
            tr(
                "Auto scales network width with the square root of the real training rows "
                "(500 rows = preset width, clipped to ×0.5–×3). Fixed always uses the "
                "preset width times the size factor."
            ),
        )
        self.model_multiplier_label = self._form_row(
            form, tr("Size factor"), self.model_multiplier_spin,
            tr("Multiplies the network width. In Auto mode it multiplies the automatic size."),
        )
        form.addRow(self.model_size_hint)
        self.loss_label = self._form_row(
            form, tr("Loss functions"), loss_row,
            tr(
                "How a network measures its error while training. Each extra tick trains "
                "every network once more. Huber is a good choice for noisy measurements."
            ),
        )
        layout.addWidget(self.neural_settings)

        self._preset_guard = False
        self.run_preset_combo.currentIndexChanged.connect(self._apply_run_preset)
        self.epochs_spin.valueChanged.connect(self._mark_preset_custom)
        self.patience_spin.valueChanged.connect(self._mark_preset_custom)
        self.model_size_combo.currentIndexChanged.connect(self._refresh_settings_visibility)
        self.model_multiplier_spin.valueChanged.connect(self._refresh_settings_visibility)
        return box

    def _build_preparation_box(self) -> QWidget:
        box = QGroupBox(tr("4 · Scaling and repeats"))
        form = self._compact_form()
        box.setLayout(form)
        scaler_row = self._plain_container()
        scaler_layout = QHBoxLayout(scaler_row)
        scaler_layout.setContentsMargins(0, 0, 0, 0)
        scaler_help = {
            "standard": tr(
                "Standard: centre on the training mean and divide by the standard "
                "deviation. Suits most models."
            ),
            "robust": tr(
                "Robust: centre on the median and divide by the interquartile range, so "
                "outliers matter less."
            ),
            "minmax": tr(
                "MinMax: squeeze into 0–1 using the training minimum and maximum. "
                "Sensitive to extreme values."
            ),
        }
        for scaler_id, label in (("standard", "Standard"), ("robust", "Robust"), ("minmax", "MinMax")):
            check = QCheckBox(label)
            check.setChecked(scaler_id == "standard")
            self._tip(check, scaler_help[scaler_id])
            check.toggled.connect(self._refresh_settings_visibility)
            self.scaler_checks[scaler_id] = check
            scaler_layout.addWidget(check)
        scaler_layout.addStretch(1)
        self._form_row(
            form, tr("Scaling"), scaler_row,
            tr(
                "How inputs and targets are rescaled, learned from the training part of "
                "each fold only. Every ticked option is compared as its own configuration."
            ),
        )
        self.repetitions_spin = self._int_spin(1, 1000, 1, suffix=" ×")
        self.repetitions_spin.valueChanged.connect(self._refresh_settings_visibility)
        self._form_row(
            form, tr("Repeat each model"), self.repetitions_spin,
            tr(
                "Trains every configuration again with a different random start and pools "
                "the predictions. Reduces luck for random models; multiplies the run time."
            ),
        )
        return box

    def _build_augmentation_section(self) -> QWidget:
        section = CollapsibleSection(tr("5 · Synthetic training rows (optional)"), expanded=False)
        self.augmentation_section = section
        outer = QVBoxLayout()
        outer.setSpacing(4)
        section.setContentLayout(outer)
        top = self._compact_form()
        self.augmentation_policy_combo = QComboBox()
        self.augmentation_policy_combo.addItem(tr("Off — train on real rows only"), "original_only")
        self.augmentation_policy_combo.addItem(
            tr("Compare each method against no augmentation"), "compare_separately"
        )
        self.augmentation_policy_combo.addItem(tr("Combine the ticked methods"), "combined")
        self._form_row(
            top, tr("Mode"), self.augmentation_policy_combo,
            tr(
                "Synthetic rows are made only from the training part of each fold, so they "
                "can never leak into the score. 'Compare' trains one extra configuration per "
                "method; 'Combine' trains one with all methods together."
            ),
        )
        outer.addLayout(top)

        self.augmentation_details = self._plain_container()
        details = QVBoxLayout(self.augmentation_details)
        details.setContentsMargins(0, 0, 0, 0)
        method_grid = QGridLayout()
        method_grid.setHorizontalSpacing(12)
        method_grid.setVerticalSpacing(2)
        methods = (
            ("c_mixup", tr("C-Mixup (target-aware)")),
            ("calibrated_noise", tr("Calibrated feature noise")),
            ("spectral_perturbation", tr("Spectral perturbation (needs a valid wavelength axis)")),
            ("feature_masking", tr("Training-time feature masking")),
            ("foma", "FOMA"),
        )
        augmentation_texts = augmentation_help()
        for index, (method_id, label) in enumerate(methods):
            check = QCheckBox(label)
            self._tip(check, augmentation_texts[method_id])
            check.toggled.connect(self._refresh_settings_visibility)
            self.augmentation_checks[method_id] = check
            method_grid.addWidget(check, index // 2, index % 2)
        details.addLayout(method_grid)
        form = self._compact_form()
        self.augmentation_form = form
        self.augmentation_ratio_spin = self._float_spin(0.05, 5.0, 0.50, step=0.25)
        self.augmentation_repeats_spin = self._int_spin(1, 100, 1, suffix=" ×")
        self.augmentation_include_original_check = QCheckBox(
            tr("Keep the real rows in the training set")
        )
        self.augmentation_include_original_check.setChecked(True)
        self.c_mixup_alpha_spin = self._float_spin(0.05, 20.0, 2.0, step=0.5)
        self.noise_fraction_spin = self._float_spin(0.0, 1.0, 0.02, step=0.01, decimals=4)
        self.feature_mask_probability_spin = self._float_spin(0.0, 0.95, 0.05, step=0.05, decimals=3)
        self.foma_alpha_spin = self._float_spin(0.05, 20.0, 2.0, step=0.5)
        self.foma_k_spin = self._int_spin(1, 10_000, 1)
        self._form_row(
            form, tr("Synthetic rows per real row"), self.augmentation_ratio_spin,
            tr("0.5 adds about half as many synthetic rows as there are training rows."),
        )
        self._form_row(
            form, tr("Multiply that by"), self.augmentation_repeats_spin,
            tr(
                "Repeats the synthetic-row generation. More rows and more time; this is not "
                "the same as repeating a model."
            ),
        )
        self._tip(
            self.augmentation_include_original_check,
            tr(
                "Ticked: train on real plus synthetic rows. Cleared: synthetic rows only. "
                "The no-augmentation baseline is always trained as well."
            ),
        )
        form.addRow(self.augmentation_include_original_check)
        self._form_row(
            form, tr("C-Mixup alpha"), self.c_mixup_alpha_spin,
            tr(
                "Shape of the mixing weight between two rows. Larger mixes closer to the "
                "middle; smaller stays close to one of the two rows."
            ),
        )
        self._form_row(
            form, tr("Noise level"), self.noise_fraction_spin,
            tr("Noise as a fraction of each feature's standard deviation."),
        )
        self._form_row(
            form, tr("Mask probability"), self.feature_mask_probability_spin,
            tr("Chance that each feature is blanked to its mean in a synthetic row."),
        )
        self._form_row(
            form, tr("FOMA alpha"), self.foma_alpha_spin,
            tr("Beta shape of the scaling applied to the minor joint X/y directions."),
        )
        self._form_row(
            form, tr("FOMA kept components (k)"), self.foma_k_spin,
            tr("How many leading joint X/y directions FOMA leaves untouched."),
        )
        details.addLayout(form)
        details.addWidget(
            self._hint(
                tr(
                    "Synthetic rows cannot add information the real rows do not have. They "
                    "only help when they encode a real measurement tolerance."
                )
            )
        )
        outer.addWidget(self.augmentation_details)
        self.augmentation_policy_combo.currentIndexChanged.connect(
            self._refresh_settings_visibility
        )
        return section

    def _build_output_box(self) -> QWidget:
        box = QGroupBox(tr("6 · Output"))
        form = self._compact_form()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        box.setLayout(form)
        folder_row = self._plain_container()
        out_row = QHBoxLayout(folder_row)
        out_row.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_ROOT))
        choose = QPushButton(tr("Browse"))
        choose.setObjectName("secondaryBtn")
        choose.clicked.connect(self.choose_output_folder)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(choose)
        self._form_row(
            form, tr("Report folder"), folder_row,
            tr(
                "Each run creates its own run_<date>_<time> subfolder here, so earlier "
                "results are never overwritten."
            ),
        )
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText(tr("e.g. NH3 first pass (blank uses the folder name)"))
        self._form_row(
            form, tr("Run name"), self.run_name_edit,
            tr("Written into every kept model, and shown on the Model library tab."),
        )
        form.addRow(
            self._hint(
                tr(
                    "Only the top {limit} models per target are saved, and only if their R² "
                    "is above 0 (better than always predicting the mean).",
                    limit=MODEL_KEEP_LIMIT,
                )
            )
        )
        return box

    # ------------------------------------------------- settings behaviour
    def _current_strategy(self) -> EvaluationStrategy:
        for strategy, button in self.strategy_buttons.items():
            if button.isChecked():
                return strategy
        return EvaluationStrategy.NESTED_CV

    def _quick_select_models(self, mode: str) -> None:
        for model_id, check in self.model_checks.items():
            if not check.isEnabled():
                continue
            if mode == "all":
                check.setChecked(True)
            elif mode == "none":
                check.setChecked(False)
            else:
                check.setChecked(model_id in RECOMMENDED_MODEL_IDS)

    def _apply_run_preset(self) -> None:
        values = {
            "economy": (250, 20),
            "balanced": (500, 30),
            "rigorous": (2000, 60),
        }.get(self.run_preset_combo.currentData())
        if values:
            self._preset_guard = True
            self.epochs_spin.setValue(values[0])
            self.patience_spin.setValue(values[1])
            self._preset_guard = False

    def _mark_preset_custom(self, *_args: object) -> None:
        if self._preset_guard:
            return
        expected = {
            "economy": (250, 20),
            "balanced": (500, 30),
            "rigorous": (2000, 60),
        }.get(self.run_preset_combo.currentData())
        if expected and expected != (self.epochs_spin.value(), self.patience_spin.value()):
            self.run_preset_combo.blockSignals(True)
            self.run_preset_combo.setCurrentIndex(self.run_preset_combo.findData("custom"))
            self.run_preset_combo.blockSignals(False)

    def _selected_model_ids(self) -> list[str]:
        return [
            model_id
            for model_id, check in self.model_checks.items()
            if check.isEnabled() and check.isChecked()
        ]

    def _typical_training_rows(self) -> int | None:
        records = self.catalog.selected_records or self.catalog.records
        rows = max((record.valid_rows for record in records), default=0)
        if not rows:
            return None
        strategy = self._current_strategy()
        if strategy == EvaluationStrategy.HOLDOUT:
            return max(1, round(rows * (1 - self.test_spin.value() / 100.0)))
        if strategy == EvaluationStrategy.TRAIN_FINAL_ONLY:
            return rows
        folds = self.folds_spin.value()
        return max(1, round(rows * (folds - 1) / folds))

    def _update_augmentation_controls(self, *_args: object) -> None:
        """Kept for callers of the old name; visibility now lives in one place."""

        self._refresh_settings_visibility()

    def _update_strategy_controls(self, *_args: object) -> None:
        self._refresh_settings_visibility()

    def _refresh_settings_visibility(self, *_args: object) -> None:
        """Show only the settings the current choices actually use."""

        if not getattr(self, "_settings_ready", False):
            return
        strategy = self._current_strategy()
        cv = strategy in {EvaluationStrategy.NESTED_CV, EvaluationStrategy.EXTERNAL_VALIDATION}
        holdout = strategy == EvaluationStrategy.HOLDOUT
        final_only = strategy == EvaluationStrategy.TRAIN_FINAL_ONLY
        form = self.strategy_form
        form.setRowVisible(self.folds_spin, cv)
        form.setRowVisible(self.test_spin, holdout)
        form.setRowVisible(self.chronological_check, holdout)
        form.setRowVisible(self.seed_spin, True)
        form.setRowVisible(self.refit_check, not final_only)
        form.setRowVisible(self.run_confirmation_check, cv)
        self.confirmation_panel.setVisible(cv and self.run_confirmation_check.isChecked())
        self.external_panel.setVisible(strategy == EvaluationStrategy.EXTERNAL_VALIDATION)
        self.strategy_description_label.setText(self.strategy_descriptions.get(strategy, ""))

        augmentation_on = self.augmentation_policy_combo.currentData() != "original_only"
        self.confirmation_form.setRowVisible(self.confirmation_aug_check, augmentation_on)
        self.augmentation_details.setVisible(augmentation_on)
        per_method = {
            self.c_mixup_alpha_spin: "c_mixup",
            self.noise_fraction_spin: "calibrated_noise",
            self.feature_mask_probability_spin: "feature_masking",
            self.foma_alpha_spin: "foma",
            self.foma_k_spin: "foma",
        }
        for control, method in per_method.items():
            self.augmentation_form.setRowVisible(control, self.augmentation_checks[method].isChecked())

        selected = self._selected_model_ids()
        neural = [model_id for model_id in selected if model_id in NEURAL_MODEL_IDS]
        loss_models = [model_id for model_id in neural if model_id in LOSS_CONFIGURABLE_MODEL_IDS]
        names = {capability.model_id: capability.display_name for capability in model_registry()}
        self.neural_settings.setVisible(bool(neural))
        if neural:
            shown = ", ".join(names.get(model_id, model_id) for model_id in neural[:4])
            if len(neural) > 4:
                shown += tr(" and {count} more", count=len(neural) - 4)
            self.neural_status_label.setText(tr("Used by: {models}", models=shown))
        else:
            self.neural_status_label.setText(
                tr("No neural network is ticked, so none of these settings are used.")
            )
        self.neural_form.setRowVisible(self.loss_label, bool(loss_models))
        auto = self.model_size_combo.currentData() == "auto"
        self.model_multiplier_label.setText(tr("Extra size factor") if auto else tr("Size factor"))
        rows = self._typical_training_rows()
        if auto and rows:
            scale = adaptive_capacity_scale(rows) * self.model_multiplier_spin.value()
            self.model_size_hint.setText(
                tr(
                    "With about {rows} training rows, networks use ×{scale:.2f} of the preset width.",
                    rows=rows,
                    scale=scale,
                )
            )
        elif auto:
            self.model_size_hint.setText(
                tr("The width is set per fold from the number of training rows.")
            )
        else:
            self.model_size_hint.setText(
                tr("Every fold uses ×{scale:.2f} of the preset width.", scale=self.model_multiplier_spin.value())
            )
        self.model_count_label.setText(
            tr("{count} selected", count=len(selected))
        )
        self._refresh_group_status()
        self._update_run_estimate()

    def _update_run_estimate(self) -> None:
        if not getattr(self, "_settings_ready", False):
            return
        records = self.catalog.selected_records
        models = self._selected_model_ids()
        if not records:
            self.run_estimate_label.setText(
                tr("Generate the combinations on the Data tab to see how large this run is.")
            )
            return
        if not models:
            self.run_estimate_label.setText(tr("Tick at least one model."))
            return
        try:
            config = self.collect_config()
        except Exception as exc:  # incomplete settings: say why instead of guessing
            self.run_estimate_label.setText(tr("Settings incomplete: {reason}", reason=str(exc)))
            return
        if config.strategy in {EvaluationStrategy.NESTED_CV, EvaluationStrategy.EXTERNAL_VALIDATION}:
            splits = config.screening_folds
        else:
            splits = 1
        fits = 0
        for record in records:
            compatible = [
                model_id
                for model_id in models
                if model_incompatibility_reason(
                    model_id,
                    record.input_columns,
                    record.feature_structure.value,
                    record.structure_confirmed,
                )
                is None
            ]
            fits += evaluation_run_count(compatible, config, splits, record)
        text = tr(
            "This run trains about {fits:,} models: {combos} combination(s) × {models} model(s) "
            "× settings × {splits} split(s).",
            fits=fits,
            combos=len(records),
            models=len(models),
            splits=splits,
        )
        if config.run_confirmation:
            text += " " + tr("The double-check stage adds many more.")
        self.run_estimate_label.setText(text)

    # ---------------------------------------------------------- grouping
    def _group_column_name(self) -> str | None:
        for (sheet, column), combo in self.column_role_boxes.items():
            if combo.currentData() == "group_id":
                return column
        return None

    def _refresh_group_status(self) -> None:
        if not hasattr(self, "group_status_label"):
            return
        if self.bundle is None:
            text, name = tr("Load data on the Data tab first."), "hint"
        else:
            column = self._group_column_name()
            if column:
                text = tr(
                    "✓ Rows are grouped by “{column}”: rows of one group are never split "
                    "between training and testing.",
                    column=column,
                )
                name = "ok"
            else:
                text = tr(
                    "⚠ No Group ID. If several rows come from the same day, tank or subject, "
                    "set that column as Group ID on the Data tab — otherwise the score is "
                    "too optimistic."
                )
                name = "warning"
        for label in (self.group_status_label, getattr(self, "data_group_status_label", None)):
            if label is None:
                continue
            label.setText(text)
            label.setObjectName(name)
            label.style().unpolish(label)
            label.style().polish(label)

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
            self.external_dataset_label.setText(tr("No separate file loaded yet"))
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
                + (
                    " "
                    + tr(
                        "⚠ {count} data check(s) flagged in the Audit column.",
                        count=self.audit_flag_count,
                    )
                    if self.audit_flag_count
                    else ""
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
        self.audit_flag_count = 0
        for sheet_name, loaded in self.bundle.sheets.items():
            profile = loaded.profile
            parent = QTreeWidgetItem((sheet_name, "", "Sheet", str(profile.cleaned_rows), str(profile.total_missing_cells), f"header={profile.header_detected}"))
            self.role_tree.addTopLevelItem(parent)
            sheet_combo = QComboBox()
            for value in ("skip", "input", "combined_input", "output", "input_or_output"):
                sheet_combo.addItem(roles[value], value)
            self.role_tree.setItemWidget(parent, 1, sheet_combo)
            self.sheet_role_boxes[sheet_name] = sheet_combo
            notes = self._column_audit_notes(loaded)
            for column_name, column in profile.column_profiles.items():
                audit_text = f"unique={column.unique_values}; numeric={column.numeric_ratio:.1%}"
                column_notes = notes.get(column_name, [])
                if column_notes:
                    audit_text = "⚠ " + "; ".join(column_notes) + " | " + audit_text
                child = QTreeWidgetItem((
                    column_name,
                    "",
                    tr("numeric") if column.trainable else tr("ID / text candidate"),
                    str(profile.cleaned_rows),
                    str(column.missing),
                    audit_text,
                ))
                if column_notes:
                    child.setForeground(5, QColor("#f5a623"))
                    child.setToolTip(5, "\n".join(column_notes))
                    self.audit_flag_count += len(column_notes)
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
        for column, width in enumerate((190, 185, 70, 50, 60)):
            self.role_tree.setColumnWidth(column, width)
        self._refresh_group_status()

    @staticmethod
    def _column_audit_notes(loaded: Any) -> dict[str, list[str]]:
        """Translated data-health notes per column (see regression_v4.data_audit)."""

        trainable = [
            name for name, profile in loaded.profile.column_profiles.items() if profile.trainable
        ]
        if not trainable:
            return {}
        notes: dict[str, list[str]] = {}
        for finding in audit_columns(loaded.data[trainable]):
            if finding.kind == "outliers":
                text = tr(
                    "{count} extreme value(s), e.g. {value:g} — typo or another population?",
                    count=finding.count,
                    value=finding.value,
                )
            elif finding.kind == "constant":
                text = tr("constant column — carries no information")
            elif finding.kind == "multiple_of":
                text = tr(
                    "exactly {value:.4g} × {other} — the same information",
                    value=finding.value,
                    other=finding.other,
                )
            else:
                text = tr(
                    "ρ = {value:.2f} with {other} — nearly the same information",
                    value=finding.value,
                    other=finding.other,
                )
            notes.setdefault(finding.column, []).append(text)
        return notes

    def _apply_identity_suggestions(self) -> None:
        """Conservatively protect obvious ID columns from accidental feature use."""

        if self.bundle is None:
            return
        observation_tokens = {"id", "編號", "编号", "樣本id", "样本id", "sampleid", "observationid"}
        group_tokens = {"groupid", "subjectid", "個體id", "个体id", "受試者id", "受试者id"}
        suggested_observation = False
        explicit_group = any(
            "".join(character.lower() for character in column if character.isalnum()) in group_tokens
            for (_sheet, column) in self.column_role_boxes
        )
        suggested_group = False
        if not explicit_group:
            # An explicit group column wins; otherwise a repeated date / batch does.
            suggested_group = self._suggest_repeated_measure_group()
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

    def _suggest_repeated_measure_group(self) -> bool:
        """Make a repeated date / batch / tank column the Group ID.

        Several rows measured on the same day share weather, biomass and the
        same calibration; scored with a random split they leak into each other
        and inflate R². Grouping by that column keeps each day on one side.
        """

        assert self.bundle is not None
        for (sheet, column), combo in self.column_role_boxes.items():
            normalized = "".join(character.lower() for character in column if character.isalnum())
            if normalized not in REPEATED_MEASURE_TOKENS:
                continue
            column_profile = self.bundle.sheets[sheet].profile.column_profiles[column]
            repeated = 1 < column_profile.unique_values < column_profile.non_missing
            index = combo.findData("group_id")
            if repeated and index >= 0 and column_profile.missing == 0:
                combo.setCurrentIndex(index)
                combo.setToolTip(
                    tr(
                        "Suggested as the Group ID: several rows share the same value, so "
                        "they are not independent. Rows of one group are kept together "
                        "when splitting. Set it back to Skip if every row is independent."
                    )
                )
                return True
        return False

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
        self._refresh_group_status()

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
        self._refresh_settings_visibility()
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

    def _selected_losses(self) -> tuple[str, ...]:
        """Ticked losses; a run without a loss-configurable network needs none."""

        ticked = tuple(key for key, check in self.loss_checks.items() if check.isChecked())
        uses_losses = any(
            model_id in LOSS_CONFIGURABLE_MODEL_IDS for model_id in self._selected_model_ids()
        )
        return ticked if ticked or uses_losses else ("mse",)

    def collect_config(self) -> EvaluationConfig:
        strategy = self._current_strategy()
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
            model_size_mode=self.model_size_combo.currentData(),
            holdout_chronological=(
                strategy == EvaluationStrategy.HOLDOUT and self.chronological_check.isChecked()
            ),
            repetitions=self.repetitions_spin.value(),
            validation_ratio=self.validation_spin.value() / 100.0,
            selected_scalers=tuple(key for key, check in self.scaler_checks.items() if check.isChecked()),
            selected_losses=self._selected_losses(),
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

    from app.wheel_guard import install_wheel_guard

    i18n.initialize()
    app = QApplication(list(argv or []))
    install_wheel_guard(app)
    window = RegressionV4MainWindow()
    window.show()
    return app.exec()
