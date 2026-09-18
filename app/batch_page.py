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

from project_paths import REPORTS_ROOT, ensure_folders

from .adapters import load_predictor
from .batch import BatchResult, load_table, run_batch, suggest_mapping
from .batch_report import BatchReportArtifacts, write_report
from .model_library import ModelEntry
from .theme import C


_NO_COLUMN = "（不對應）"


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
                            error=f"載入失敗：{type(exc).__name__}: {exc}",
                        )
                    )
                    self.log.emit(f"[{entry.display_name}] 載入失敗：{exc}")
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
                        f"[{entry.display_name}] {result.used_rows} 列：R²={metrics.reported_r2:.4f}, "
                        f"MAE={metrics.mae:.6g}, MAPE={metrics.mape_percent:.2f}%, "
                        f"NMAE={metrics.nmae:.6g}"
                    )
                else:
                    self.log.emit(f"[{entry.display_name}] {result.used_rows} 列預測完成（無 ground truth）")

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

        source = QGroupBox("1. 資料來源")
        source_layout = QVBoxLayout(source)
        row = QHBoxLayout()
        browse = QPushButton("選擇 CSV / Excel")
        browse.clicked.connect(self.browse_table)
        self.path_label = QLabel("尚未載入資料表")
        self.path_label.setStyleSheet(f"color: {C['muted']};")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setMinimumWidth(180)
        self.sheet_combo.currentIndexChanged.connect(self._on_sheet_changed)
        self.observation_combo = QComboBox()
        self.observation_combo.setMinimumWidth(180)
        row.addWidget(browse)
        row.addWidget(self.path_label, 1)
        row.addWidget(QLabel("工作表："))
        row.addWidget(self.sheet_combo)
        row.addWidget(QLabel("觀測 ID 欄："))
        row.addWidget(self.observation_combo)
        source_layout.addLayout(row)
        self.table_summary = QLabel("載入後會顯示列數與欄位數。")
        self.table_summary.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        source_layout.addWidget(self.table_summary)
        layout.addWidget(source)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        models_box = QGroupBox("2. 選擇模型（可多選比較）")
        models_layout = QVBoxLayout(models_box)
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.model_list.itemChanged.connect(self._on_model_selection_changed)
        models_layout.addWidget(self.model_list, 1)
        select_row = QHBoxLayout()
        select_all = QPushButton("全選")
        select_all.setObjectName("secondaryBtn")
        select_all.clicked.connect(lambda: self._set_all_models(True))
        select_none = QPushButton("全不選")
        select_none.setObjectName("secondaryBtn")
        select_none.clicked.connect(lambda: self._set_all_models(False))
        select_row.addWidget(select_all)
        select_row.addWidget(select_none)
        select_row.addStretch(1)
        models_layout.addLayout(select_row)
        splitter.addWidget(models_box)

        mapping_box = QGroupBox("3. 欄位對應（自動比對，可手動修正）")
        mapping_layout = QVBoxLayout(mapping_box)
        self.mapping_model_combo = QComboBox()
        self.mapping_model_combo.currentIndexChanged.connect(self._show_mapping)
        mapping_layout.addWidget(self.mapping_model_combo)
        self.mapping_table = QTableWidget(0, 3)
        self.mapping_table.setHorizontalHeaderLabels(("用途", "模型欄位名稱", "資料表欄位"))
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
            "把模型的輸出欄位也對應到資料表，就會把這份資料當作 test data 驗證並輸出比對報告。"
        )
        self.mapping_hint.setWordWrap(True)
        self.mapping_hint.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        mapping_layout.addWidget(self.mapping_hint)
        splitter.addWidget(mapping_box)
        splitter.setSizes([420, 700])
        layout.addWidget(splitter, 1)

        run_box = QGroupBox("4. 執行與結果")
        run_layout = QVBoxLayout(run_box)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("▶ 執行批次推論")
        self.run_button.clicked.connect(self.start_batch)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("secondaryBtn")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.open_report_button = QPushButton("開啟報告資料夾")
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
        self.result_tabs.addTab(self.metrics_table, "驗證指標")
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.verticalHeader().setVisible(False)
        self.result_tabs.addTab(self.preview_table, "預測結果預覽")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.result_tabs.addTab(self.log_box, "執行紀錄")
        run_layout.addWidget(self.result_tabs, 1)
        layout.addWidget(run_box, 1)

    # -------------------------------------------------------------- loading
    def browse_table(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "選擇批次推論資料表", str(Path.home()), "Data (*.xlsx *.xls *.csv)"
        )
        if path:
            self.load_table(path)

    def load_table(self, path: str) -> None:
        try:
            self.bundle = load_table(path)
        except Exception as exc:
            QMessageBox.critical(self, "資料載入失敗", f"{type(exc).__name__}: {exc}")
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
            f"工作表「{sheet}」：{len(self.frame):,} 列 × {len(columns)} 欄"
        )
        self.observation_combo.blockSignals(True)
        self.observation_combo.clear()
        self.observation_combo.addItem(_NO_COLUMN)
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
                f"{entry.path}\n輸入：{'、'.join(entry.feature_names) or '—'}\n"
                f"輸出：{'、'.join(entry.target_names) or '—'}"
            )
            self.model_list.addItem(item)
        self.model_list.blockSignals(False)
        if not self.entries:
            self.mapping_hint.setText(
                "目前沒有已勾選且可執行的模型。請先到「模型庫」分頁勾選要使用的模型。"
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

        rows = [("輸入", name, mapping["features"].get(name)) for name in entry.feature_names]
        rows += [("輸出", name, mapping["targets"].get(name)) for name in entry.target_names]
        self.mapping_table.setRowCount(len(rows))
        for row_index, (role, name, selected) in enumerate(rows):
            role_item = QTableWidgetItem(role)
            role_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.mapping_table.setItem(row_index, 0, role_item)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setToolTip(name)
            self.mapping_table.setItem(row_index, 1, name_item)

            combo = QComboBox()
            combo.addItem(_NO_COLUMN)
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
                f"⚠️ 還有 {len(unmapped_features)} 個輸入欄位未對應："
                f"{'、'.join(unmapped_features[:5])}{'…' if len(unmapped_features) > 5 else ''}"
            )
        elif mapped_targets == len(entry.target_names) and entry.target_names:
            self.mapping_hint.setText(
                "✅ 輸入與全部 ground truth 欄位都已對應：這次會輸出驗證比對報告。"
            )
        else:
            self.mapping_hint.setText(
                "輸入已對應完成。把輸出欄位也對應到資料表，就會把這份資料當作 test data 驗證。"
            )

    def _set_mapping(self, entry_id: str, role: str, name: str, combo: QComboBox) -> None:
        key = "features" if role == "輸入" else "targets"
        value = combo.currentText()
        bucket = self.mappings.setdefault(entry_id, {"features": {}, "targets": {}})[key]
        if value == _NO_COLUMN:
            bucket.pop(name, None)
        else:
            bucket[name] = value
        self._show_mapping()

    # ------------------------------------------------------------ execution
    def start_batch(self) -> None:
        if self.frame is None:
            QMessageBox.warning(self, "尚未載入資料", "請先選擇要批次推論的 CSV 或 Excel 檔。")
            return
        chosen = self.selected_entries()
        if not chosen:
            QMessageBox.warning(self, "尚未選擇模型", "請至少勾選一個模型。")
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
                "欄位對應不完整",
                "這些模型還有未對應的輸入欄位，執行後會被標記為失敗：\n\n"
                + "\n".join(incomplete[:8])
                + "\n\n仍要繼續嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        observation = self.observation_combo.currentText()
        observation_column = None if observation == _NO_COLUMN else observation
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
            self.log_box.appendPlainText("已送出取消請求；目前模型完成後停止。")

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
        self.log_box.appendPlainText(f"報告已輸出：{artifacts.folder}")
        if artifacts.pdf_error:
            self.log_box.appendPlainText(f"PDF 產生失敗（其他檔案不受影響）：{artifacts.pdf_error}")
        validated = sum(1 for result in results if result.has_validation)
        QMessageBox.information(
            self,
            "批次推論完成",
            f"{len(results)} 個模型完成，其中 {validated} 個有 ground truth 可驗證。\n\n"
            f"報告資料夾：\n{artifacts.folder}",
        )

    def _on_failed(self, details: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.log_box.appendPlainText(details)
        QMessageBox.critical(self, "批次推論失敗", details[-3000:])

    def _fill_metrics(self, results: list[BatchResult]) -> None:
        rows = [row for result in results for row in result.metric_rows()]
        if not rows:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(1)
            self.metrics_table.setHorizontalHeaderLabels(("訊息",))
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(
                0, 0, QTableWidgetItem("沒有 ground truth 欄位，因此沒有驗證指標。")
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
            QMessageBox.warning(self, "無法開啟資料夾", str(exc))

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
