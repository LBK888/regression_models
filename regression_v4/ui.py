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


ROLE_OPTIONS = {
    "skip": "略過",
    "input": "輸入",
    "combined_input": "合併輸入",
    "output": "輸出",
    "input_or_output": "可作輸入或輸出",
    "observation_id": "Observation ID（不作為特徵）",
    "group_id": "Group ID（不作為特徵）",
}

HELP_GLYPH = "⍰"


MODEL_HELP = {
    "mean": "平均值基準模型：只用訓練資料的目標平均值預測；勾選後可判斷複雜模型是否真的優於最簡單基準。",
    "ridge": "Ridge 線性迴歸：以 L2 正則化縮小不穩定係數；勾選後會評估較抗共線性的線性關係。",
    "pls": "PLS：把高度相關特徵壓縮成少數與目標相關的潛在成分；適合特徵多且共線的小資料。",
    "svr_rbf": "RBF-SVR：以核函數學習非線性關係並控制誤差帶；勾選後會增加一組非線性小樣本比較。",
    "random_forest": "Random Forest：對多棵隨機決策樹取平均；可描述非線性與交互作用，但訓練時間較長。",
    "extra_trees": "Extra Trees：使用更隨機的切分建立樹集成；通常較快並可降低單棵樹的變異。",
    "deep_mlp": "Deep MLP：多層全連接神經網路；會學習一般非線性關係並使用所選 neural loss。",
    "wide_mlp": "Wide MLP：較寬的全連接神經網路；增加同層表示能力，也會增加參數與運算量。",
    "residual_mlp": "Residual MLP：以殘差連接訓練較深網路；可改善梯度傳遞並增加非線性容量。",
    "multi_branch_mlp": "Multi-branch MLP：以多條並行網路路徑抽取不同表示後合併；會增加模型容量與運算量。",
    "bottleneck_mlp": "Bottleneck MLP：先壓縮再重建高階表示；可促使模型學到較精簡的特徵組合。",
    "cnn1d": "CNN1D：沿使用者確認的有序一維特徵滑動卷積核；只適合相鄰欄位確實具有局部關係的資料。",
    "resnet1d": "ResNet1D：在有序一維特徵上使用殘差卷積區塊；適合序列型資料，不要求光譜命名。",
    "grouped_fusion": "Grouped Fusion：每個 Feature Group 使用獨立編碼器再融合；只有至少兩個有意義群組時才執行。",
    "mlp_embeddings": "Numerical-Embedding MLP：先把每個數值特徵轉成可學習表示再交給 MLP；可捕捉一般表格非線性。",
    "ft_transformer": "FT-Transformer：把每個特徵視為 token 並用 attention 建模特徵互動；適合一般表格資料但較耗時。",
    "modern_nca": "ModernNCA：學習讓相似目標彼此接近的嵌入空間，再以鄰居預測；適合探索局部樣本結構。",
    "tabm": "TabM：以參數共享的多成員表格神經網路集成預測；套件可用時才會執行。",
    "realmlp": "RealMLP：使用 pytabkit 的表格 MLP 設定；套件可用時加入比較。",
    "catboost": "CatBoost：梯度提升決策樹；擅長一般表格非線性，安裝 optional package 後才可使用。",
    "xgboost": "XGBoost：正則化梯度提升樹；勾選後加入常用表格 boosting 比較，運算量會增加。",
    "lightgbm": "LightGBM：以 leaf-wise 策略建立梯度提升樹；通常速度快，安裝 optional package 後才可使用。",
}

LOSS_HELP = {
    "mse": "MSE：平方較大的誤差，因此會更重視離群的大偏差；勾選後 neural models 會多訓練一組 MSE configuration。",
    "huber": "Huber／SmoothL1：小誤差用平方、大誤差改用近似線性；可降低離群值對 neural training 的影響。",
    "mae": "MAE／L1：所有絕對誤差按比例計算；較不受離群值影響，但梯度較不平滑。",
    "log_cosh": "Log-cosh：小誤差近似 MSE、大誤差近似 MAE；提供平滑且較耐離群值的折衷。",
}

AUGMENTATION_HELP = {
    "c_mixup": "C-Mixup：依目標值相近程度挑選兩筆訓練資料並內插 X 與 y；勾選後會建立 target-aware synthetic rows。",
    "calibrated_noise": "Calibrated noise：依訓練特徵離散程度對 X 加入小幅隨機雜訊，y 保持來源值；用來模擬量測變動。",
    "spectral_perturbation": "Spectral perturbation：沿有效波長軸加入平滑基線、倍率與雜訊變化；非光譜 Experiment 會自動略過。",
    "feature_masking": "Feature masking：訓練時隨機遮蔽部分已縮放特徵；迫使模型不要過度依賴單一欄位。",
    "foma": "FOMA：對 fold-training 的聯合 X/y 做 SVD 並縮放非主要流形成分；用來產生流形附近的 synthetic rows。",
}


class DatasetDropFrame(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        label = QLabel("將 CSV / Excel 檔拖曳到這裡，或按下方按鈕選擇")
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
    HEADERS = (
        "使用", "ID", "Inputs", "Outputs", "有效列", "輸入欄數", "輸出欄數",
        "Stable Key", "Target Task", "Feature Structure", "結構確認",
    )

    def __init__(self, catalog: ExperimentCatalog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.catalog = catalog

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.catalog.records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
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
                return "取消勾選即可讓這個組合不參與訓練；重新產生組合時會依 Stable Key 保留選擇。"
            if column == 9:
                return (
                    f"Feature Groups: {', '.join(record.feature_groups)}\n"
                    "可修改自動建議；CNN1D/ResNet1D/Grouped Fusion 只會執行已確認且相容的組合。"
                )
            if column == 10:
                return "確認 Feature Structure 與實際資料語意一致後再勾選。"
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
                    self.cancelled.emit("已取消訓練。")
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
                    self.log_message.emit("  警告：未指定 Observation ID，目前使用列位置作為對齊備援。")
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
            self.cancelled.emit("已取消訓練。")
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
        self._role_change_guard = False
        self.setting_help_labels: list[QLabel] = []
        self.setting_help_controls: list[QWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "1. 資料與組合")
        self.tabs.addTab(self._build_config_tab(), "2. 評估設定")
        self.tabs.addTab(self._build_run_tab(), "3. 執行與報告")
        layout.addWidget(self.tabs)

    def _build_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        drop = DatasetDropFrame()
        drop.file_dropped.connect(self.load_dataset)
        layout.addWidget(drop)
        row = QHBoxLayout()
        browse = QPushButton("選擇資料檔")
        browse.clicked.connect(self.browse_dataset)
        self.dataset_label = QLabel("尚未載入資料")
        preview = QPushButton("產生／更新排列組合")
        preview.clicked.connect(self.generate_preview)
        row.addWidget(browse)
        row.addWidget(self.dataset_label, 1)
        row.addWidget(preview)
        layout.addLayout(row)
        self.dataset_summary = QLabel("Observation ID 與 Group ID 不會進入模型特徵。")
        self.dataset_summary.setWordWrap(True)
        layout.addWidget(self.dataset_summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.role_tree = QTreeWidget()
        self.role_tree.setColumnCount(6)
        self.role_tree.setHeaderLabels(("Sheet / 欄位", "角色", "型態", "列數", "缺失", "稽核資訊"))
        splitter.addWidget(self.role_tree)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        tools = QHBoxLayout()
        self.combo_label = QLabel("尚未產生組合")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("篩選 ID、Inputs、Outputs、Stable Key…")
        all_button = QPushButton("全選")
        none_button = QPushButton("全不選")
        invert_button = QPushButton("反選")
        confirm_structures_button = QPushButton("確認已選組合的結構建議")
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
            "決定如何產生泛化證據；這不是參數 preset，切換策略只會停用不適用欄位並保留原數值。"
        )
        form = QFormLayout(evaluation)
        self.strategy_combo = QComboBox()
        strategy_rows = (
            (
                "兩階段 Benchmark：CV Screening + optional Confirmation",
                EvaluationStrategy.NESTED_CV,
                "以 development data 做 K-fold Screening，可選 repeated nested Confirmation；忽略 Test ratio 與 External mapping。",
            ),
            (
                "Holdout：單次 Train/Test split",
                EvaluationStrategy.HOLDOUT,
                "依 Test ratio 隨機切出一次測試集；忽略 Screening folds、Confirmation 與 External mapping。",
            ),
            (
                "External Validation：獨立第二資料集",
                EvaluationStrategy.EXTERNAL_VALIDATION,
                "先用 development CV 選定設定，再只對映射後的外部資料評估一次；忽略 Test ratio。",
            ),
            (
                "Train Final Model Only：只訓練、不評估",
                EvaluationStrategy.TRAIN_FINAL_ONLY,
                "全部 development rows 直接建立 final model；忽略 Test ratio、folds、Confirmation 與 External mapping，不產生泛化指標。",
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
            "Evaluation Strategy 決定證據流程，不會像 Run preset 一樣改寫 epochs 等數值；不適用設定會停用並在執行時忽略。",
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
        self.refit_check = QCheckBox("評估完成後，以全部資料重新 fit final model")
        self.refit_check.setChecked(True)
        self.run_confirmation_check = QCheckBox("Screening 後執行 repeated nested Confirmation")
        self.run_confirmation_check.setChecked(False)
        self.finalists_spin = QSpinBox(); self.finalists_spin.setRange(1, 20); self.finalists_spin.setValue(5)
        self.confirmation_outer_folds_spin = QSpinBox(); self.confirmation_outer_folds_spin.setRange(2, 10); self.confirmation_outer_folds_spin.setValue(5)
        self.confirmation_outer_repeats_spin = QSpinBox(); self.confirmation_outer_repeats_spin.setRange(1, 20); self.confirmation_outer_repeats_spin.setValue(3)
        self.confirmation_inner_folds_spin = QSpinBox(); self.confirmation_inner_folds_spin.setRange(2, 10); self.confirmation_inner_folds_spin.setValue(3)
        self.tuning_budget_spin = QSpinBox(); self.tuning_budget_spin.setRange(1, 100); self.tuning_budget_spin.setValue(8)
        self.confirmation_aug_check = QCheckBox("以相同 outer folds 比較 Original 與已選 augmentation")
        self.confirmation_aug_check.setChecked(True)
        self._add_help_row(
            form, "策略", self.strategy_combo,
            "選擇泛化證據的產生方式；它不是 preset，不會改寫下方數值，只會使用或忽略對應欄位。",
        )
        self._add_help_row(
            form, "Test ratio", self.test_spin,
            "只供 Holdout 使用：依百分比分出一次測試資料；設為 0% 時全部資料用於 final fit，但不產生測試證據。",
        )
        self._add_help_row(
            form, "Screening folds", self.folds_spin,
            "把 development data 輪流切成 K 份，每次用一份評估、其餘訓練；K 越大評估次數與時間越多。",
        )
        self._add_help_row(
            form, "Random seed", self.seed_spin,
            "控制資料切分、初始化與 augmentation 的隨機序列；相同資料與設定使用相同 seed 可重現結果。",
        )
        self._help_control(
            self.refit_check,
            "評估結束並選定設定後，以全部 development rows 重新訓練可保存的 final model；這個 fit 不會產生泛化證據。",
            append_question=True,
        )
        form.addRow(self.refit_check)
        self._help_control(
            self.run_confirmation_check,
            "將 Screening 晉級的候選模型再做 repeated nested CV；結果較可靠但運算量顯著增加，只適用 CV 型策略。",
            append_question=True,
        )
        form.addRow(self.run_confirmation_check)
        self._add_help_row(
            form, "Finalists / Target Task", self.finalists_spin,
            "每個相同輸出目標最多晉級多少候選設定到 Confirmation；數量越多，確認時間越長。",
        )
        self._add_help_row(
            form, "Confirmation outer folds", self.confirmation_outer_folds_spin,
            "Confirmation 外層 CV 的切分數；每個 outer test fold 都完全不參與該次調參與訓練。",
        )
        self._add_help_row(
            form, "Confirmation outer repeats", self.confirmation_outer_repeats_spin,
            "用不同 seed 重複整套 outer CV；可估計結果變異，但運算量近似按次數倍增。",
        )
        self._add_help_row(
            form, "Confirmation inner folds", self.confirmation_inner_folds_spin,
            "只在每個 outer training partition 內做 inner CV 選參數；可避免用 outer test data 調參。",
        )
        self._add_help_row(
            form, "Tuning budget / finalist", self.tuning_budget_spin,
            "每個 finalist 在每個 outer fold 最多嘗試的參數組數；數值越大搜尋更廣但耗時增加。",
        )
        self._help_control(
            self.confirmation_aug_check,
            "在完全相同的 outer folds 上比較 Original 與啟用的 augmentation；Original only 時此項會被忽略。",
            append_question=True,
        )
        form.addRow(self.confirmation_aug_check)
        self.strategy_description_label = QLabel()
        self.strategy_description_label.setWordWrap(True)
        self.strategy_description_label.setStyleSheet("color: #315a7d; padding: 4px;")
        form.addRow(self.strategy_description_label)
        note = QLabel("提示：灰色欄位代表目前策略不使用；切換策略後原輸入值仍會保留。")
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

        models_box = QGroupBox(f"模型（灰色項目刻意保留，以免後續遺漏） {HELP_GLYPH}")
        models_box.setToolTip("勾選要比較的模型；每增加一個模型都會依 scaler、loss、augmentation 與 repetitions 展開更多訓練。")
        models_layout = QGridLayout(models_box)
        for index, capability in enumerate(model_registry()):
            check = QCheckBox(capability.display_name)
            check.setEnabled(capability.enabled)
            check.setChecked(capability.model_id in {"mean", "ridge", "pls", "svr_rbf"})
            if capability.availability == ModelAvailability.COMING_LATER:
                check.setText(f"{capability.display_name} — Coming Later")
            elif not capability.enabled:
                check.setText(f"{capability.display_name} — Inactive")
            model_explanation = MODEL_HELP.get(
                capability.model_id,
                "勾選後會在每個相容 Experiment 中加入此模型比較；模型越多，總運算時間越長。",
            )
            if not capability.enabled:
                model_explanation += f" 目前不可執行：{capability.reason}"
            self._help_control(check, model_explanation, append_question=True)
            self.model_checks[capability.model_id] = check
            models_layout.addWidget(check, index // 2, index % 2)
        layout.addWidget(models_box, 0, 1)

        advanced = QGroupBox(f"進階訓練設定（Preset 只會填入預設值，所有欄位仍可修改） {HELP_GLYPH}")
        advanced.setToolTip("控制 neural training、重複次數與特徵縮放；Run preset 是唯一會立即改寫部分數值的 preset。")
        advanced_form = QFormLayout(advanced)
        self.run_preset_combo = QComboBox()
        self.run_preset_combo.addItem("Economy", "economy")
        self.run_preset_combo.addItem("Balanced", "balanced")
        self.run_preset_combo.addItem("Rigorous", "rigorous")
        self.run_preset_combo.addItem("Custom", "custom")
        preset_help = {
            "economy": "快速預覽：降低 epochs、patience 與 repetitions，較省時間但結果穩定性較低。",
            "balanced": "平衡模式：使用中等 epochs、patience 與 3 次 repetitions，兼顧時間與穩定性。",
            "rigorous": "嚴謹模式：提高 epochs、patience、模型寬度與 5 次 repetitions，運算量大幅增加。",
            "custom": "自訂模式：不改寫任何數值，保留目前手動設定。",
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
            "standard": "Standard scaler：以 training fold 的平均值與標準差轉成約為零均值、單位尺度；適合多數模型。",
            "robust": "Robust scaler：以中位數與四分位距縮放；較不受離群值影響。",
            "minmax": "MinMax scaler：依 training fold 最小與最大值縮放到固定範圍；對極端值較敏感。",
        }
        for scaler_id, label in (("standard", "Standard"), ("robust", "Robust"), ("minmax", "MinMax")):
            check = QCheckBox(label)
            check.setChecked(scaler_id == "standard")
            self._help_control(check, scaler_help[scaler_id], append_question=True)
            self.scaler_checks[scaler_id] = check
            scaler_layout.addWidget(check)
        self._add_help_row(
            advanced_form, "Run preset", self.run_preset_combo,
            "真正的參數 preset：切換 Economy／Balanced／Rigorous 會立即改寫 epochs、patience、repetitions、validation ratio 與 model multiplier；其他欄位不變。",
        )
        self._add_help_row(
            advanced_form, "Epochs", self.epochs_spin,
            "Neural model 最多完整掃過 training data 的次數；較大可能學得更充分，但會增加時間並可能過擬合。",
        )
        self._add_help_row(
            advanced_form, "Early-stopping patience", self.patience_spin,
            "Validation loss 連續多少 epochs 沒改善才停止；較大會等待更久，較小可能過早停止。",
        )
        self._add_help_row(
            advanced_form, "Batch size", self.batch_spin,
            "每次梯度更新使用的 training rows 數；較大通常較快但耗記憶體，較小更新較有隨機性。",
        )
        self._add_help_row(
            advanced_form, "Learning rate", self.learning_rate_spin,
            "Neural optimizer 每次更新權重的步幅；太大可能不穩定，太小會收斂緩慢。",
        )
        self._add_help_row(
            advanced_form, "Model width multiplier", self.model_multiplier_spin,
            "按倍率放大或縮小 neural hidden width；較大增加模型容量、記憶體與過擬合風險。",
        )
        self._add_help_row(
            advanced_form, "Training repetitions", self.repetitions_spin,
            "用不同隨機初始化重複每個 configuration 並彙整預測；可降低偶然性，但時間近似按倍數增加。",
        )
        self._add_help_row(
            advanced_form, "Internal validation ratio", self.validation_spin,
            "只從 neural model 的 fold-training partition 再切一部分監控 early stopping；不會取用 fold test data。",
        )
        self._help_control(
            scaler_row,
            "勾選一種或多種 fold-local scaling；每多一種 scaler 都會增加一整組 model configurations。",
        )
        advanced_form.addRow(
            self._help_label(
                "Scalers",
                "控制每個 training fold 如何縮放 X 與 y；每個勾選 scaler 都會分開訓練與評估。",
            ),
            scaler_row,
        )
        self.run_preset_combo.currentIndexChanged.connect(self._apply_run_preset)
        layout.addWidget(advanced, 1, 0)

        losses = QGroupBox(f"Loss functions（只套用於支援自訂 loss 的 neural adapters） {HELP_GLYPH}")
        losses.setToolTip("Loss 決定 neural model 訓練時如何衡量預測誤差；每多勾一種就增加一組 neural configuration。")
        loss_layout = QVBoxLayout(losses)
        for loss_id, label, checked in (
            ("mse", "MSE", True),
            ("huber", "Huber / SmoothL1（建議用於 noisy measurements）", True),
            ("mae", "MAE / L1", False),
            ("log_cosh", "Log-cosh", False),
        ):
            check = QCheckBox(label)
            check.setChecked(checked)
            self._help_control(check, LOSS_HELP[loss_id], append_question=True)
            self.loss_checks[loss_id] = check
            loss_layout.addWidget(check)
        loss_note = QLabel("Classical models 使用其原生 objective，不會因勾選多個 loss 而重複執行。")
        loss_note.setWordWrap(True); loss_layout.addWidget(loss_note)
        layout.addWidget(losses, 1, 1)

        augmentation = CollapsibleSection(f"Augmentation policy {HELP_GLYPH}", expanded=False)
        augmentation.toggle_button.setToolTip(
            "決定 synthetic training rows 如何加入比較；Original only 會忽略所有已勾選 methods。"
        )
        self.augmentation_section = augmentation
        aug_layout = QFormLayout()
        augmentation.setContentLayout(aug_layout)
        self.augmentation_policy_combo = QComboBox()
        self.augmentation_policy_combo.addItem("Original only（不做 augmentation）", "original_only")
        self.augmentation_policy_combo.addItem("與原始資料分開比較（建議）", "compare_separately")
        self.augmentation_policy_combo.addItem("合併已勾選方法", "combined")
        policy_help = {
            "original_only": "只建立 Original configuration；methods 的勾選狀態會保留但執行時完全忽略。",
            "compare_separately": "建立 Original 與每個已勾選 method 的獨立 configuration，可直接判斷各方法是否改善結果。",
            "combined": "建立 Original 與一個合併所有已勾選 methods 的 configuration；可測組合效果但無法分辨單一方法貢獻。",
        }
        for index in range(self.augmentation_policy_combo.count()):
            self.augmentation_policy_combo.setItemData(
                index,
                policy_help[self.augmentation_policy_combo.itemData(index)],
                Qt.ItemDataRole.ToolTipRole,
            )
        self._add_help_row(
            aug_layout, "Policy", self.augmentation_policy_combo,
            "控制 augmentation configurations 的展開方式；Original only 不做 augmentation，Separate 分開比較，Combined 合併方法。",
        )
        method_widget = QWidget(); method_layout = QGridLayout(method_widget); method_layout.setContentsMargins(0, 0, 0, 0)
        methods = (
            ("c_mixup", "C-Mixup（target-aware）"),
            ("calibrated_noise", "Calibrated feature noise"),
            ("spectral_perturbation", "Spectral perturbation（需有效波長軸）"),
            ("feature_masking", "Training-time feature masking"),
            ("foma", "FOMA (First-Order Manifold Data Augmentation)"),
        )
        for index, (method_id, label) in enumerate(methods):
            check = QCheckBox(label)
            self._help_control(check, AUGMENTATION_HELP[method_id], append_question=True)
            self.augmentation_checks[method_id] = check
            method_layout.addWidget(check, index // 2, index % 2)
        self._help_control(
            method_widget,
            "選擇要產生的 training-only augmentation；是否分開或合併由 Policy 決定。",
        )
        aug_layout.addRow(
            self._help_label(
                "Methods",
                "勾選 augmentation 演算法；Original only 時全部忽略，Separate 時各自比較，Combined 時一起套用。",
            ),
            method_widget,
        )
        self.augmentation_repeats_spin = QSpinBox(); self.augmentation_repeats_spin.setRange(1, 100); self.augmentation_repeats_spin.setValue(1)
        self.augmentation_ratio_spin = QDoubleSpinBox(); self.augmentation_ratio_spin.setRange(0.05, 5.0); self.augmentation_ratio_spin.setSingleStep(0.25); self.augmentation_ratio_spin.setValue(0.50)
        self.augmentation_include_original_check = QCheckBox("將原始資料加入 augmentation 訓練集")
        self.augmentation_include_original_check.setChecked(True)
        self.c_mixup_alpha_spin = QDoubleSpinBox(); self.c_mixup_alpha_spin.setRange(0.05, 20.0); self.c_mixup_alpha_spin.setValue(2.0)
        self.noise_fraction_spin = QDoubleSpinBox(); self.noise_fraction_spin.setDecimals(4); self.noise_fraction_spin.setRange(0.0, 1.0); self.noise_fraction_spin.setValue(0.02)
        self.feature_mask_probability_spin = QDoubleSpinBox(); self.feature_mask_probability_spin.setDecimals(3); self.feature_mask_probability_spin.setRange(0.0, 0.95); self.feature_mask_probability_spin.setValue(0.05)
        self.foma_alpha_spin = QDoubleSpinBox(); self.foma_alpha_spin.setRange(0.05, 20.0); self.foma_alpha_spin.setValue(2.0)
        self.foma_k_spin = QSpinBox(); self.foma_k_spin.setRange(1, 10_000); self.foma_k_spin.setValue(1)
        self._add_help_row(
            aug_layout, "Synthetic rows / train rows", self.augmentation_ratio_spin,
            "每次 augmentation 產生的 synthetic rows 相對於 fold-training rows 的比例；0.5 表示產生約一半數量。",
        )
        self._add_help_row(
            aug_layout, "Augmentation repeats", self.augmentation_repeats_spin,
            "把 synthetic row 數量再乘上此次數；增加資料量與訓練時間，但不是模型 training repetitions。",
        )
        self._help_control(
            self.augmentation_include_original_check,
            "勾選時 augmented variant 使用 Original rows 加 synthetic rows；取消時該 variant 只使用 synthetic rows，Original baseline 仍保留。",
            append_question=True,
        )
        aug_layout.addRow(self.augmentation_include_original_check)
        self._add_help_row(
            aug_layout, "C-Mixup alpha", self.c_mixup_alpha_spin,
            "Beta 分布的形狀參數，控制兩筆資料內插權重；較大更接近中間混合，較小更靠近其中一筆。",
        )
        self._add_help_row(
            aug_layout, "Noise fraction", self.noise_fraction_spin,
            "Calibrated noise 相對於 training feature 標準差的比例；越大擾動越強，也越可能偏離合理量測範圍。",
        )
        self._add_help_row(
            aug_layout, "Mask probability", self.feature_mask_probability_spin,
            "每個 training feature 被遮蔽為縮放後平均值的機率；越大正則化越強，也可能損失過多資訊。",
        )
        self._add_help_row(
            aug_layout, "FOMA alpha", self.foma_alpha_spin,
            "控制 FOMA 非主要奇異成分的 Beta 縮放係數；較大偏向中等縮放，較小較常接近保留或大幅壓縮。",
        )
        self._add_help_row(
            aug_layout, "FOMA retained components (k)", self.foma_k_spin,
            "FOMA 完整保留的前 k 個聯合 X/y SVD 成分；k 越大資料主結構保留越多、擾動越保守。",
        )
        self.augmentation_policy_description_label = QLabel()
        self.augmentation_policy_description_label.setWordWrap(True)
        self.augmentation_policy_description_label.setStyleSheet("color: #315a7d; padding: 4px;")
        aug_layout.addRow(self.augmentation_policy_description_label)
        aug_note = QLabel(
            "所有方法只作用於當前 fold 的 training partition，並保留 Original baseline。"
            "Bootstrap 不列為 augmentation；SMOGN 必須先定義 target relevance，因此本版不提供任意開關。"
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
            "只供 External Validation 策略使用：將 development 欄位逐一對應到獨立外部資料，模型選定後才讀取外部 target。"
        )
        self.external_validation_section = external
        external_layout = QVBoxLayout()
        external.setContentLayout(external_layout)
        external_top = QHBoxLayout()
        external_button = QPushButton(f"載入獨立外部資料 {HELP_GLYPH}")
        self._help_control(
            external_button,
            "載入完全獨立、未參與 development selection 的資料檔；只在 External Validation 策略執行一次最終評估。",
        )
        external_button.clicked.connect(self.browse_external_dataset)
        self.external_dataset_label = QLabel("尚未載入；只在 External Validation 策略使用")
        external_top.addWidget(external_button)
        external_top.addWidget(self.external_dataset_label, 1)
        external_layout.addLayout(external_top)
        self.external_observation_combo = QComboBox()
        self.external_observation_combo.addItem("使用外部資料列位置", None)
        external_layout.addWidget(self._help_label(
            "External Observation ID（選填）",
            "選擇外部資料中可唯一識別每筆觀測的欄位；未指定時用資料列位置對齊並在報告中警告。",
        ))
        self._help_control(
            self.external_observation_combo,
            "指定外部資料的觀測識別欄；只用於追蹤與對齊，不會當成模型輸入特徵。",
        )
        external_layout.addWidget(self.external_observation_combo)
        self.external_mapping_tree = QTreeWidget()
        self.external_mapping_tree.setColumnCount(2)
        self.external_mapping_tree.setHeaderLabels(
            (f"Development 欄位 {HELP_GLYPH}", f"External 欄位 {HELP_GLYPH}")
        )
        self._help_control(
            self.external_mapping_tree,
            "左欄是模型建立時使用的欄位，右欄選擇外部資料的對應欄位；所有 inputs 與 target 都必須明確映射。",
        )
        self.external_mapping_tree.setMinimumHeight(150)
        self.external_mapping_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        external_layout.addWidget(self.external_mapping_tree)
        external_note = QLabel(
            "必須明確映射所有入模特徵與目標。外部 target 在 development CV／Confirmation 完成並凍結勝出設定前不會被讀取。"
        )
        external_note.setWordWrap(True)
        external_layout.addWidget(external_note)
        layout.addWidget(external, 3, 0, 1, 2)

        output = QGroupBox(f"輸出 {HELP_GLYPH}")
        output.setToolTip("指定所有實驗結果、圖表、Markdown、PDF 與 final model bundles 的主輸出資料夾。")
        output_form = QFormLayout(output)
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_ROOT))
        self._help_control(
            self.output_edit,
            "這是報告主資料夾；每次開始執行都會在其中自動建立帶時間的獨立 run 子資料夾，避免覆蓋舊結果。",
        )
        choose = QPushButton(f"選擇 {HELP_GLYPH}")
        self._help_control(choose, "開啟資料夾選擇器並把選定路徑填入 Master report 資料夾。")
        choose.clicked.connect(self.choose_output_folder)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(choose)
        output_form.addRow(
            self._help_label(
                "Master report 資料夾",
                "作為所有 Benchmark Runs 的上層目錄；每次執行會建立 run_年月日_時間_微秒子資料夾，不會搬移或覆蓋先前結果。",
            ),
            out_row,
        )
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("例如：泰國蝦 qPCR 初篩（留空則用資料夾名稱）")
        self._add_help_row(
            output_form,
            "Run 名稱",
            self.run_name_edit,
            "這個名稱會寫進每個保留模型的 metadata，之後在「模型庫」頁面用來分辨不同實驗目的的訓練批次。",
        )
        keep_note = QLabel(
            f"每個 Target Task 只保留排名前 {MODEL_KEEP_LIMIT} 的模型；"
            "Reported R² 未大於 0 的模型（包含第一名）不會被儲存。"
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
            "original_only": (
                "目前只執行 Original：所有已勾選 methods 與其參數都會忽略；"
                "勾選狀態會保留，切回其他 policy 即可繼續使用。"
            ),
            "compare_separately": (
                "目前會建立 Original baseline，再將每個已勾選 method 各自建立一個 configuration；"
                "方法之間不混合。"
            ),
            "combined": (
                "目前會建立 Original baseline，再建立一個合併所有已勾選 methods 的 configuration；"
                "報告只能判斷整體組合，不能拆解單一方法貢獻。"
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
            EvaluationStrategy.NESTED_CV: (
                "這是評估工作流程，不是 preset。使用 Screening folds；可選 Confirmation。"
                "Test ratio 與 External mapping 會忽略，其他 training／model／loss／augmentation 設定照常使用。"
            ),
            EvaluationStrategy.HOLDOUT: (
                "這是評估工作流程，不是 preset。只使用 Test ratio 做一次 Train/Test split；"
                "Screening folds、所有 Confirmation 設定與 External mapping 會忽略。Test ratio=0% 時只做 final fit。"
            ),
            EvaluationStrategy.EXTERNAL_VALIDATION: (
                "這是評估工作流程，不是 preset。先使用 Screening folds 在 development data 選定設定，"
                "可選 Confirmation，最後才使用 External mapping 評估一次；Test ratio 會忽略。"
            ),
            EvaluationStrategy.TRAIN_FINAL_ONLY: (
                "這是評估工作流程，不是 preset。使用全部 development rows 直接訓練 final models；"
                "Test ratio、Screening folds、Confirmation、External mapping 與 refit 選項都會忽略，且不產生泛化指標。"
            ),
        }
        self.strategy_description_label.setText(descriptions.get(strategy, ""))

    def _build_run_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("開始執行")
        self.start_button.clicked.connect(self.start_training)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.current_run_label = QLabel("Training log — 尚未開始")
        self.current_run_label.setWordWrap(True)
        layout.addWidget(self.current_run_label)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)
        return tab

    def browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇資料檔", str(Path.cwd()), "Data (*.xlsx *.xls *.csv)")
        if path:
            self.load_dataset(path)

    def browse_external_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇獨立 External Validation 資料檔",
            str(Path.cwd()),
            "Data (*.xlsx *.xls *.csv)",
        )
        if path:
            self.load_external_dataset(path)

    def load_external_dataset(self, path: str) -> None:
        try:
            self.external_bundle = FlexibleDatasetLoader().load(path)
            self.external_dataset_label.setText(path)
            self._refresh_external_mapping()
        except Exception as exc:
            QMessageBox.critical(self, "外部資料載入失敗", str(exc))

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
        self.external_observation_combo.addItem("使用外部資料列位置", None)
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
            combo.addItem("— 請選擇 —", None)
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
            raise ValueError("External mapping 尚未完成：" + "、".join(missing))
        return mapping, self.external_observation_combo.currentData()

    def load_dataset(self, path: str) -> None:
        try:
            self.bundle = FlexibleDatasetLoader().load(path)
            self.external_bundle = None
            self.external_dataset_label.setText("尚未載入；只在 External Validation 策略使用")
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
                f"已載入 {len(self.bundle.sheets)} 個 sheets / tables，共 {rows} 列。"
                "請指定 Observation ID；若同一個體有重複量測，另指定 Group ID 以避免 split leakage。"
            )
        except Exception as exc:
            QMessageBox.critical(self, "資料載入失敗", str(exc))

    def _populate_roles(self) -> None:
        assert self.bundle is not None
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
                sheet_combo.addItem(ROLE_OPTIONS[value], value)
            self.role_tree.setItemWidget(parent, 1, sheet_combo)
            self.sheet_role_boxes[sheet_name] = sheet_combo
            for column_name, column in profile.column_profiles.items():
                child = QTreeWidgetItem((column_name, "", "數值" if column.trainable else "ID / 文字候選", str(profile.cleaned_rows), str(column.missing), f"unique={column.unique_values}; numeric={column.numeric_ratio:.1%}"))
                parent.addChild(child)
                combo = QComboBox()
                values = ["skip", "observation_id", "group_id"]
                if column.trainable:
                    values[1:1] = ["input", "combined_input", "output", "input_or_output"]
                for value in values:
                    combo.addItem(ROLE_OPTIONS[value], value)
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
                    combo.setToolTip("V4 依欄名自動建議為 Observation ID；請確認，這一欄不會進入模型特徵。")
                    suggested_observation = True
            elif not suggested_group and normalized in group_tokens:
                index = combo.findData("group_id")
                if index >= 0:
                    combo.setCurrentIndex(index)
                    combo.setToolTip("V4 依欄名自動建議為 Group ID；請確認。")
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
        self.combo_label.setText("角色已變更，請重新產生組合")

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
            QMessageBox.warning(self, "尚未載入資料", "請先載入 CSV 或 Excel。")
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
                QMessageBox.warning(self, "沒有有效組合", "至少需要一組輸入與一組輸出。")
        except Exception as exc:
            QMessageBox.critical(self, "組合產生失敗", str(exc))

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
            f"已選 {len(self.catalog.selected_records):,} / {len(self.catalog.records):,} 組；"
            f"結構已確認 {confirmed:,}"
        )

    def choose_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", self.output_edit.text())
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
            QMessageBox.warning(self, "尚未載入資料", "請先載入資料並設定角色。")
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
            QMessageBox.warning(self, "執行計畫為空", "請至少勾選一個資料組合與一個可用模型。")
            return
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "設定不完整", str(exc))
            return
        external_mapping: dict[str, ColumnRef] = {}
        external_observation_ref: ColumnRef | None = None
        if config.strategy == EvaluationStrategy.EXTERNAL_VALIDATION:
            if self.external_bundle is None:
                QMessageBox.warning(self, "尚需外部資料", "請載入獨立 External Validation 資料並完成欄位 mapping。")
                return
            try:
                external_mapping, external_observation_ref = self._collect_external_mapping(jobs)
            except Exception as exc:
                QMessageBox.warning(self, "External mapping 不完整", str(exc))
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
            self.log_box.appendPlainText("已送出取消請求；目前 fold 完成後停止。")

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
            self.current_run_label.setText(f"Training log — Completed — {artifacts.pdf_path}")
            QMessageBox.information(self, "完成", f"Master report 已輸出：\n{artifacts.pdf_path}")
        else:
            error_path = getattr(artifacts, "rendering_error_path", None)
            self.current_run_label.setText("Training log — Completed with PDF rendering error")
            QMessageBox.warning(
                self,
                "訓練完成，但 PDF 產生失敗",
                f"CSV、Markdown、PNG、模型與稽核檔均已保留。\n錯誤紀錄：{error_path}",
            )

    def on_failed(self, details: str) -> None:
        self._finish_worker()
        self.log_box.appendPlainText(details)
        QMessageBox.critical(self, "執行失敗", details[-3000:])

    def on_cancelled(self, message: str) -> None:
        self._finish_worker()
        self.current_run_label.setText(message)

    def _finish_worker(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def can_close(self) -> bool:
        """Refuse to close while a Benchmark Run still owns a worker thread."""

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "訓練進行中", "請先取消並等待目前 fold 結束。")
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
    app = QApplication(list(argv or []))
    window = RegressionV4MainWindow()
    window.show()
    return app.exec()
