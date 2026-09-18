# -*- coding: utf-8 -*-
"""Batch inference over a worksheet, with test-data validation and a report.

Pick a table, pick a sheet, pick one or more models from the library, confirm
the column mapping, and run. When ground-truth columns are mapped as well, the
sheet is treated as test data and the run produces a comparison report whose
analysis items mirror the Benchmark Run report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import numpy as np
import pandas as pd

from i18n import tr
from project_paths import REPORTS_ROOT, ensure_folders

from .adapters import load_predictor
from .batch import BatchResult, load_table, run_batch, suggest_mapping
from .batch_report import BatchReportArtifacts, write_report
from .model_library import ModelEntry
from .theme import C


def no_column_label() -> str:
    """The "leave this name unmapped" entry, resolved per build for i18n."""

    return tr("(not mapped)")


_ROLE_INPUT = "input"
_ROLE_TARGET = "target"


class BatchWorker(QThread):
    """Load each model, run it over the sheet and write the report."""

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_batch = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        entries: list[ModelEntry],
        frame: pd.DataFrame,
        mappings: dict[str, dict[str, dict[str, str]]],
        observation_column: str | None,
        output_folder: Path,
        source_path: str,
        sheet: str,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.frame = frame
        self.mappings = mappings
        self.observation_column = observation_column
        self.output_folder = output_folder
        self.source_path = source_path
        self.sheet = sheet

    def run(self) -> None:
        try:
            results: list[BatchResult] = []
            total = len(self.entries)
            for position, entry in enumerate(self.entries, start=1):
                if self.isInterruptionRequested():
                    break
                self.progress.emit(position, total, entry.display_name)
                mapping = self.mappings.get(entry.entry_id, {})
                try:
                    predictor = load_predictor(entry.path)
                except Exception as exc:
                    results.append(
                        BatchResult(
                            model_label=entry.display_name,
                            feature_names=entry.feature_names,
                            target_names=entry.target_names,
                            row_labels=(),
                            predictions=np.empty((0, 0)),
                            total_rows=len(self.frame),
                            error=tr(
                                "Loading failed: {error_type}: {error}",
                                error_type=type(exc).__name__,
                                error=str(exc),
                            ),
                        )
                    )
                    self.log.emit(
                        f"[{entry.display_name}] "
                        + tr("loading failed: {error}", error=str(exc))
                    )
                    continue
                result = run_batch(
                    predictor,
                    self.frame,
                    mapping.get("features", {}),
                    model_label=entry.display_name,
                    target_mapping=mapping.get("targets", {}),
                    observation_column=self.observation_column,
                )
                results.append(result)
                if result.error:
                    self.log.emit(f"[{entry.display_name}] {result.error}")
                elif result.has_validation:
                    metrics = result.metrics
                    self.log.emit(
                        f"[{entry.display_name}] "
                        + tr("{rows} row(s)", rows=result.used_rows)
                        + f": R²={metrics.reported_r2:.4f}, MAE={metrics.mae:.6g}, "
                        f"MAPE={metrics.mape_percent:.2f}%, NMAE={metrics.nmae:.6g}"
                    )
                else:
                    self.log.emit(
                        f"[{entry.display_name}] "
                        + tr(
                            "predicted {rows} row(s); no ground truth to validate against",
                            rows=result.used_rows,
                        )
                    )

            artifacts = write_report(
                results,
                self.output_folder,
                source_path=self.source_path,
                sheet=self.sheet,
                total_rows=len(self.frame),
                mappings={
                    entry.display_name: self.mappings.get(entry.entry_id, {})
                    for entry in self.entries
                },
            )
            self.finished_batch.emit(results, artifacts)
        except Exception as exc:
            import traceback

            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class BatchPage(QWidget):
    """The batch inference and validation workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ensure_folders()
        self.entries: list[ModelEntry] = []
        self.bundle = None
        self.frame: pd.DataFrame | None = None
        self.mappings: dict[str, dict[str, dict[str, str]]] = {}
        self.worker: BatchWorker | None = None
        self.last_artifacts: BatchReportArtifacts | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self.source_box = QGroupBox(tr("1. Data source"))
        source_layout = QVBoxLayout(self.source_box)
        row = QHBoxLayout()
        self.browse_button = QPushButton(tr("Choose a CSV / Excel file"))
        self.browse_button.clicked.connect(self.browse_table)
        self.path_label = QLabel(tr("No table loaded yet"))
        self.path_label.setStyleSheet(f"color: {C['muted']};")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setMinimumWidth(180)
        self.sheet_combo.currentIndexChanged.connect(self._on_sheet_changed)
        self.observation_combo = QComboBox()
        self.observation_combo.setMinimumWidth(180)
        self.sheet_caption = QLabel(tr("Worksheet:"))
        self.observation_caption = QLabel(tr("Observation ID column:"))
        row.addWidget(self.browse_button)
        row.addWidget(self.path_label, 1)
        row.addWidget(self.sheet_caption)
        row.addWidget(self.sheet_combo)
        row.addWidget(self.observation_caption)
        row.addWidget(self.observation_combo)
        source_layout.addLayout(row)
        self.table_summary = QLabel(tr("Row and column counts appear once a table is loaded."))
        self.table_summary.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        source_layout.addWidget(self.table_summary)
        layout.addWidget(self.source_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.models_box = QGroupBox(tr("2. Models (tick several to compare them)"))
        models_layout = QVBoxLayout(self.models_box)
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.model_list.itemChanged.connect(self._on_model_selection_changed)
        models_layout.addWidget(self.model_list, 1)
        select_row = QHBoxLayout()
        self.select_all_button = QPushButton(tr("Select all"))
        self.select_all_button.setObjectName("secondaryBtn")
        self.select_all_button.clicked.connect(lambda: self._set_all_models(True))
        self.select_none_button = QPushButton(tr("Select none"))
        self.select_none_button.setObjectName("secondaryBtn")
        self.select_none_button.clicked.connect(lambda: self._set_all_models(False))
        select_row.addWidget(self.select_all_button)
        select_row.addWidget(self.select_none_button)
        select_row.addStretch(1)
        models_layout.addLayout(select_row)
        splitter.addWidget(self.models_box)

        self.mapping_box = QGroupBox(tr("3. Column mapping (matched automatically, editable)"))
        mapping_layout = QVBoxLayout(self.mapping_box)
        self.mapping_model_combo = QComboBox()
        self.mapping_model_combo.currentIndexChanged.connect(self._show_mapping)
        mapping_layout.addWidget(self.mapping_model_combo)
        self.mapping_table = QTableWidget(0, 3)
        self.mapping_table.setHorizontalHeaderLabels(
            (tr("Role"), tr("Model column name"), tr("Table column"))
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.mapping_table.setColumnWidth(0, 90)
        self.mapping_table.verticalHeader().setVisible(False)
        mapping_layout.addWidget(self.mapping_table, 1)
        self.mapping_hint = QLabel(
            tr(
                "Map the model's output columns as well and this table is treated as "
                "test data: the run then produces a validation comparison report."
            )
        )
        self.mapping_hint.setWordWrap(True)
        self.mapping_hint.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        mapping_layout.addWidget(self.mapping_hint)
        splitter.addWidget(self.mapping_box)
        splitter.setSizes([420, 700])
        layout.addWidget(splitter, 1)

        self.run_box = QGroupBox(tr("4. Run and results"))
        run_layout = QVBoxLayout(self.run_box)
        buttons = QHBoxLayout()
        self.run_button = QPushButton(tr("▶ Run batch inference"))
        self.run_button.clicked.connect(self.start_batch)
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setObjectName("secondaryBtn")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.open_report_button = QPushButton(tr("Open the report folder"))
        self.open_report_button.setObjectName("secondaryBtn")
        self.open_report_button.setEnabled(False)
        self.open_report_button.clicked.connect(self.open_report)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.open_report_button)
        buttons.addStretch(1)
        run_layout.addLayout(buttons)
        self.progress = QProgressBar()
        run_layout.addWidget(self.progress)

        self.result_tabs = QTabWidget()
        self.metrics_table = QTableWidget(0, 0)
        self.metrics_table.verticalHeader().setVisible(False)
        self.result_tabs.addTab(self.metrics_table, tr("Validation metrics"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.verticalHeader().setVisible(False)
        self.result_tabs.addTab(self.preview_table, tr("Prediction preview"))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.result_tabs.addTab(self.log_box, tr("Run log"))
        run_layout.addWidget(self.result_tabs, 1)
        layout.addWidget(self.run_box, 1)

    def retranslate(self) -> None:
        """Re-apply every static string after a language change."""

        self.source_box.setTitle(tr("1. Data source"))
        self.browse_button.setText(tr("Choose a CSV / Excel file"))
        self.sheet_caption.setText(tr("Worksheet:"))
        self.observation_caption.setText(tr("Observation ID column:"))
        self.models_box.setTitle(tr("2. Models (tick several to compare them)"))
        self.select_all_button.setText(tr("Select all"))
        self.select_none_button.setText(tr("Select none"))
        self.mapping_box.setTitle(tr("3. Column mapping (matched automatically, editable)"))
        self.mapping_table.setHorizontalHeaderLabels(
            (tr("Role"), tr("Model column name"), tr("Table column"))
        )
        self.run_box.setTitle(tr("4. Run and results"))
        self.run_button.setText(tr("▶ Run batch inference"))
        self.cancel_button.setText(tr("Cancel"))
        self.open_report_button.setText(tr("Open the report folder"))
        self.result_tabs.setTabText(0, tr("Validation metrics"))
        self.result_tabs.setTabText(1, tr("Prediction preview"))
        self.result_tabs.setTabText(2, tr("Run log"))
        if self.frame is None:
            self.path_label.setText(tr("No table loaded yet"))
            self.table_summary.setText(
                tr("Row and column counts appear once a table is loaded.")
            )
        else:
            self._on_sheet_changed(self.sheet_combo.currentIndex())
        self.set_entries(self.entries)

    # -------------------------------------------------------------- loading
    def browse_table(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            tr("Choose the table to run inference on"),
            str(Path.home()),
            tr("Data ({patterns})", patterns="*.xlsx *.xls *.csv"),
        )
        if path:
            self.load_table(path)

    def load_table(self, path: str) -> None:
        try:
            self.bundle = load_table(path)
        except Exception as exc:
            QMessageBox.critical(
                self, tr("Could not load the table"), f"{type(exc).__name__}: {exc}"
            )
            return
        self.path_label.setText(path)
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        for name in self.bundle.sheets:
            self.sheet_combo.addItem(name)
        self.sheet_combo.blockSignals(False)
        self._on_sheet_changed(0)

    def _on_sheet_changed(self, _index: int) -> None:
        if self.bundle is None or not self.sheet_combo.count():
            return
        sheet = self.sheet_combo.currentText()
        self.frame = self.bundle.sheets[sheet].data
        columns = [str(column) for column in self.frame.columns]
        self.table_summary.setText(
            tr(
                "Worksheet “{sheet}”: {rows:,} rows × {columns} columns",
                sheet=sheet,
                rows=len(self.frame),
                columns=len(columns),
            )
        )
        self.observation_combo.blockSignals(True)
        self.observation_combo.clear()
        self.observation_combo.addItem(no_column_label())
        for column in columns:
            self.observation_combo.addItem(column)
        self.observation_combo.blockSignals(False)
        self._rebuild_mappings()

    # --------------------------------------------------------------- models
    def set_entries(self, entries: list[ModelEntry]) -> None:
        checked = {
            self.model_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.model_list.count())
            if self.model_list.item(index).checkState() == Qt.CheckState.Checked
        }
        self.entries = list(entries)
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for entry in self.entries:
            item = QListWidgetItem(f"{entry.run_label} / {entry.summary_line()}")
            item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if entry.entry_id in checked else Qt.CheckState.Unchecked
            )
            item.setToolTip(
                f"{entry.path}\n"
                + tr("Inputs: {names}", names=", ".join(entry.feature_names) or "—")
                + "\n"
                + tr("Outputs: {names}", names=", ".join(entry.target_names) or "—")
            )
            self.model_list.addItem(item)
        self.model_list.blockSignals(False)
        if not self.entries:
            self.mapping_hint.setText(
                tr(
                    "No model is both ticked and runnable. Tick the ones you want on the "
                    "Model library tab first."
                )
            )
        self._rebuild_mappings()

    def _set_all_models(self, checked: bool) -> None:
        self.model_list.blockSignals(True)
        for index in range(self.model_list.count()):
            self.model_list.item(index).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self.model_list.blockSignals(False)
        self._rebuild_mappings()

    def _on_model_selection_changed(self, _item: QListWidgetItem) -> None:
        self._rebuild_mappings()

    def selected_entries(self) -> list[ModelEntry]:
        chosen: list[ModelEntry] = []
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            entry_id = item.data(Qt.ItemDataRole.UserRole)
            entry = next((candidate for candidate in self.entries if candidate.entry_id == entry_id), None)
            if entry is not None:
                chosen.append(entry)
        return chosen

    # -------------------------------------------------------------- mapping
    def _rebuild_mappings(self) -> None:
        chosen = self.selected_entries()
        columns = [str(column) for column in (self.frame.columns if self.frame is not None else [])]
        for entry in chosen:
            if entry.entry_id in self.mappings:
                continue
            self.mappings[entry.entry_id] = {
                "features": {
                    name: column
                    for name, column in suggest_mapping(entry.feature_names, columns).items()
                    if column
                },
                "targets": {
                    name: column
                    for name, column in suggest_mapping(entry.target_names, columns).items()
                    if column
                },
            }
        for entry_id in list(self.mappings):
            if entry_id not in {entry.entry_id for entry in chosen}:
                self.mappings.pop(entry_id)

        previous = self.mapping_model_combo.currentData()
        self.mapping_model_combo.blockSignals(True)
        self.mapping_model_combo.clear()
        for entry in chosen:
            self.mapping_model_combo.addItem(entry.display_name, entry.entry_id)
        index = self.mapping_model_combo.findData(previous) if previous else 0
        self.mapping_model_combo.setCurrentIndex(max(0, index))
        self.mapping_model_combo.blockSignals(False)
        self._show_mapping()

    def remap_all(self) -> None:
        """Recompute every suggestion, e.g. after switching sheets."""

        self.mappings.clear()
        self._rebuild_mappings()

    def _show_mapping(self) -> None:
        entry_id = self.mapping_model_combo.currentData()
        self.mapping_table.setRowCount(0)
        if entry_id is None or self.frame is None:
            return
        entry = next((item for item in self.entries if item.entry_id == entry_id), None)
        if entry is None:
            return
        columns = [str(column) for column in self.frame.columns]
        mapping = self.mappings.setdefault(entry_id, {"features": {}, "targets": {}})

        rows = [
            (_ROLE_INPUT, name, mapping["features"].get(name)) for name in entry.feature_names
        ]
        rows += [
            (_ROLE_TARGET, name, mapping["targets"].get(name)) for name in entry.target_names
        ]
        role_labels = {_ROLE_INPUT: tr("Input"), _ROLE_TARGET: tr("Output")}
        self.mapping_table.setRowCount(len(rows))
        for row_index, (role, name, selected) in enumerate(rows):
            role_item = QTableWidgetItem(role_labels[role])
            role_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.mapping_table.setItem(row_index, 0, role_item)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setToolTip(name)
            self.mapping_table.setItem(row_index, 1, name_item)

            combo = QComboBox()
            combo.addItem(no_column_label())
            for column in columns:
                combo.addItem(column)
            combo.setCurrentIndex(columns.index(selected) + 1 if selected in columns else 0)
            combo.currentIndexChanged.connect(
                lambda _index, e=entry_id, r=role, n=name, box=combo: self._set_mapping(e, r, n, box)
            )
            self.mapping_table.setCellWidget(row_index, 2, combo)

        mapped_targets = sum(1 for name in entry.target_names if mapping["targets"].get(name))
        unmapped_features = [name for name in entry.feature_names if not mapping["features"].get(name)]
        if unmapped_features:
            self.mapping_hint.setText(
                tr(
                    "⚠️ {count} input column(s) are still unmapped: {names}",
                    count=len(unmapped_features),
                    names=", ".join(unmapped_features[:5])
                    + ("…" if len(unmapped_features) > 5 else ""),
                )
            )
        elif mapped_targets == len(entry.target_names) and entry.target_names:
            self.mapping_hint.setText(
                tr(
                    "✅ Inputs and every ground-truth column are mapped: this run "
                    "will produce a validation comparison report."
                )
            )
        else:
            self.mapping_hint.setText(
                tr(
                    "Inputs are fully mapped. Map the output columns too and this table "
                    "is treated as test data."
                )
            )

    def _set_mapping(self, entry_id: str, role: str, name: str, combo: QComboBox) -> None:
        key = "features" if role == _ROLE_INPUT else "targets"
        value = combo.currentText()
        bucket = self.mappings.setdefault(entry_id, {"features": {}, "targets": {}})[key]
        if value == no_column_label():
            bucket.pop(name, None)
        else:
            bucket[name] = value
        self._show_mapping()

    # ------------------------------------------------------------ execution
    def start_batch(self) -> None:
        if self.frame is None:
            QMessageBox.warning(
                self,
                tr("No table loaded"),
                tr("Choose the CSV or Excel file to run inference on first."),
            )
            return
        chosen = self.selected_entries()
        if not chosen:
            QMessageBox.warning(
                self, tr("No model selected"), tr("Tick at least one model.")
            )
            return
        incomplete = [
            entry.display_name
            for entry in chosen
            if any(not self.mappings.get(entry.entry_id, {}).get("features", {}).get(name)
                   for name in entry.feature_names)
        ]
        if incomplete:
            answer = QMessageBox.question(
                self,
                tr("Incomplete column mapping"),
                tr(
                    "These models still have unmapped input columns and will be reported "
                    "as failed:\n\n{names}\n\nRun anyway?",
                    names="\n".join(incomplete[:8]),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        observation = self.observation_combo.currentText()
        observation_column = None if observation == no_column_label() else observation
        folder = REPORTS_ROOT / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.log_box.clear()
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = BatchWorker(
            chosen,
            self.frame,
            self.mappings,
            observation_column,
            folder,
            self.path_label.text(),
            self.sheet_combo.currentText(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.finished_batch.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def cancel_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.log_box.appendPlainText(
                tr("Cancellation requested; the run stops after the current model.")
            )

    def _on_progress(self, current: int, total: int, label: str) -> None:
        self.progress.setValue(round(100 * current / max(1, total)))
        self.log_box.appendPlainText(f"[{current}/{total}] {label}")

    def _on_finished(self, results: list[BatchResult], artifacts: BatchReportArtifacts) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setValue(100)
        self.last_artifacts = artifacts
        self.open_report_button.setEnabled(True)
        self._fill_metrics(results)
        self._fill_preview(results)
        self.log_box.appendPlainText(
            tr("Report written to: {path}", path=str(artifacts.folder))
        )
        if artifacts.pdf_error:
            self.log_box.appendPlainText(
                tr(
                    "The PDF could not be produced (every other file is intact): {error}",
                    error=artifacts.pdf_error,
                )
            )
        validated = sum(1 for result in results if result.has_validation)
        QMessageBox.information(
            self,
            tr("Batch inference finished"),
            tr(
                "{total} model(s) finished, {validated} of them with ground truth to "
                "validate against.\n\nReport folder:\n{path}",
                total=len(results),
                validated=validated,
                path=str(artifacts.folder),
            ),
        )

    def _on_failed(self, details: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.log_box.appendPlainText(details)
        QMessageBox.critical(self, tr("Batch inference failed"), details[-3000:])

    def _fill_metrics(self, results: list[BatchResult]) -> None:
        rows = [row for result in results for row in result.metric_rows()]
        if not rows:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(1)
            self.metrics_table.setHorizontalHeaderLabels((tr("Message"),))
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(
                0,
                0,
                QTableWidgetItem(
                    tr("No ground-truth column was mapped, so there are no validation metrics.")
                ),
            )
            self.metrics_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            return
        frame = pd.DataFrame(rows)
        self._fill_table(self.metrics_table, frame)

    def _fill_preview(self, results: list[BatchResult]) -> None:
        frames = [result.wide_frame().assign(model=result.model_label) for result in results if result.used_rows]
        if not frames:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
        frame = pd.concat(frames, ignore_index=True).head(500)
        self._fill_table(self.preview_table, frame)

    @staticmethod
    def _fill_table(table: QTableWidget, frame: pd.DataFrame) -> None:
        table.setRowCount(len(frame))
        table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        for row_index, (_index, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.6g}"
                else:
                    text = "" if value is None else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()

    def open_report(self) -> None:
        if self.last_artifacts is None:
            return
        import os

        try:
            os.startfile(str(self.last_artifacts.folder))  # type: ignore[attr-defined]
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not open the folder"), str(exc))

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
