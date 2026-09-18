# -*- coding: utf-8 -*-
"""The model library tab: name runs and models, and choose what stays visible.

Training is run repeatedly and for different purposes, so the inference tabs
must not simply list every bundle on disk. This page is where the user decides:
it groups models by Benchmark Run, shows each model's Reported R² and MAPE,
lets runs and models be renamed, and lets each model be ticked in or out of the
inference tabs. Models that cannot be executed stay listed with the reason they
are incompatible instead of disappearing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from project_paths import MODEL_LIBRARY_ROOT, ensure_folders

from .adapters import MODEL_SUFFIXES
from .model_library import LibraryIndex, ModelEntry, build_entry, discover_model_files, probe_entry
from .theme import C


COLUMNS = (
    "名稱 / Run",
    "使用",
    "模型",
    "R²",
    "MAPE %",
    "輸入",
    "輸出",
    "類型",
    "狀態",
)

_ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1
_RUN_ROLE = Qt.ItemDataRole.UserRole + 2


class ScanWorker(QThread):
    """Discover and probe models off the UI thread."""

    entry_found = pyqtSignal(object)
    progress = pyqtSignal(int, int, str)
    finished_scan = pyqtSignal(int)

    def __init__(self, index: LibraryIndex) -> None:
        super().__init__()
        self.index = index

    def run(self) -> None:
        paths = discover_model_files()
        total = len(paths)
        for position, path in enumerate(paths, start=1):
            if self.isInterruptionRequested():
                break
            self.progress.emit(position, total, path.name)
            entry = build_entry(path, self.index)
            entry, _predictor = probe_entry(entry)
            self.entry_found.emit(entry)
        self.finished_scan.emit(total)


class LibraryPage(QWidget):
    """Model management, and the single source of "which models are visible"."""

    library_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ensure_folders()
        self.index = LibraryIndex()
        self.entries: dict[str, ModelEntry] = {}
        self.worker: ScanWorker | None = None
        self._updating = False
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("模型庫")
        title.setStyleSheet(f"color: {C['accent2']}; font-size: 18px; font-weight: bold;")
        self.status_label = QLabel("尚未掃描")
        self.status_label.setStyleSheet(f"color: {C['muted']};")
        header.addWidget(title)
        header.addWidget(self.status_label, 1)

        self.rescan_button = QPushButton("🔄 重新掃描")
        self.rescan_button.setObjectName("secondaryBtn")
        self.rescan_button.clicked.connect(self.refresh)
        self.import_button = QPushButton("➕ 匯入模型檔")
        self.import_button.setObjectName("secondaryBtn")
        self.import_button.clicked.connect(self.import_models)
        header.addWidget(self.import_button)
        header.addWidget(self.rescan_button)
        layout.addLayout(header)

        hint = QLabel(
            "在這裡為 run 與模型命名、勾選要在推論頁面看到的模型。"
            "第三方或舊格式模型也會列出；無法執行的會在「狀態」欄說明原因。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        layout.addWidget(hint)

        tools = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("篩選 run、模型名稱、目標或狀態…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.only_usable_check = QCheckBox("只顯示可執行的模型")
        self.only_usable_check.toggled.connect(self._apply_filter)
        select_all = QPushButton("全部勾選")
        select_all.setObjectName("secondaryBtn")
        select_all.clicked.connect(lambda: self._set_all_enabled(True))
        select_none = QPushButton("全部取消")
        select_none.setObjectName("secondaryBtn")
        select_none.clicked.connect(lambda: self._set_all_enabled(False))
        tools.addWidget(self.filter_edit, 1)
        tools.addWidget(self.only_usable_check)
        tools.addWidget(select_all)
        tools.addWidget(select_none)
        layout.addLayout(tools)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        header_view = self.tree.header()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(0, 360)
        for column in range(1, len(COLUMNS)):
            self.tree.setColumnWidth(column, 110)
        splitter.addWidget(self.tree)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        detail_layout.addWidget(QLabel("選取項目詳細資料 / 備註（備註會自動儲存）"))
        self.detail_box = QPlainTextEdit()
        self.detail_box.setReadOnly(True)
        self.detail_box.setMaximumHeight(150)
        detail_layout.addWidget(self.detail_box)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("為這個模型或 run 寫下用途說明…")
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.textChanged.connect(self._on_notes_changed)
        detail_layout.addWidget(self.notes_edit)
        splitter.addWidget(detail)
        splitter.setSizes([560, 220])
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------- scanning
    def refresh(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.index.load()
        self.entries.clear()
        self._updating = True
        self.tree.clear()
        self._updating = False
        self.rescan_button.setEnabled(False)
        self.status_label.setText("掃描中…")
        self.worker = ScanWorker(self.index)
        self.worker.entry_found.connect(self._add_entry)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_scan.connect(self._on_scan_finished)
        self.worker.start()

    def _on_progress(self, current: int, total: int, name: str) -> None:
        self.status_label.setText(f"掃描中 {current}/{total}：{name}")

    def _on_scan_finished(self, total: int) -> None:
        self.rescan_button.setEnabled(True)
        self.index.prune(self.entries.keys())
        self.index.save()
        usable = sum(1 for entry in self.entries.values() if entry.compatible)
        enabled = sum(1 for entry in self.entries.values() if entry.usable)
        self.status_label.setText(
            f"共 {total} 個模型檔；可執行 {usable} 個；已勾選供推論使用 {enabled} 個"
        )
        self._apply_filter()
        self.library_changed.emit()

    def _add_entry(self, entry: ModelEntry) -> None:
        self.entries[entry.entry_id] = entry
        self._updating = True
        try:
            run_item = self._run_item(entry.run_folder, entry.run_label)
            item = QTreeWidgetItem(run_item)
            self._fill_item(item, entry)
            run_item.setExpanded(True)
        finally:
            self._updating = False

    def _run_item(self, run_folder: str, run_label: str) -> QTreeWidgetItem:
        for index in range(self.tree.topLevelItemCount()):
            candidate = self.tree.topLevelItem(index)
            if candidate.data(0, _RUN_ROLE) == run_folder:
                return candidate
        item = QTreeWidgetItem(self.tree)
        item.setText(0, run_label)
        item.setData(0, _RUN_ROLE, run_folder)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setForeground(0, Qt.GlobalColor.white)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

    def _fill_item(self, item: QTreeWidgetItem, entry: ModelEntry) -> None:
        item.setData(0, _ENTRY_ROLE, entry.entry_id)
        item.setText(0, entry.display_name or entry.path.stem)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            1, Qt.CheckState.Checked if entry.enabled else Qt.CheckState.Unchecked
        )
        item.setText(2, entry.model_name or entry.architecture or "—")
        r2 = entry.reported_r2
        item.setText(3, f"{r2:.4f}" if r2 is not None else "—")
        mape = entry.mape_percent
        item.setText(4, f"{mape:.2f}" if mape is not None else "—")
        item.setText(5, str(len(entry.feature_names)) if entry.feature_names else "—")
        item.setText(6, str(len(entry.target_names)) if entry.target_names else "—")
        item.setText(7, entry.kind_label)
        item.setText(8, entry.reason)
        colour = C["success"] if entry.compatible else C["error"]
        item.setForeground(8, _brush(colour))
        if not entry.compatible:
            item.setForeground(0, _brush(C["muted"]))

    # -------------------------------------------------------------- editing
    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating:
            return
        run_folder = item.data(0, _RUN_ROLE)
        if run_folder is not None:
            self.index.set_run_name(str(run_folder), item.text(0))
            self.index.save()
            for entry_id, entry in list(self.entries.items()):
                if entry.run_folder == str(run_folder):
                    self.entries[entry_id] = replace(entry, run_label=item.text(0))
            self.library_changed.emit()
            return

        entry_id = item.data(0, _ENTRY_ROLE)
        if entry_id is None:
            return
        entry = self.entries.get(str(entry_id))
        if entry is None:
            return
        if column == 0:
            name = item.text(0).strip() or entry.default_name
            self.index.set_display_name(entry.entry_id, name)
            self.entries[entry.entry_id] = replace(entry, display_name=name)
        elif column == 1:
            enabled = item.checkState(1) == Qt.CheckState.Checked
            if enabled and not entry.compatible:
                self._updating = True
                item.setCheckState(1, Qt.CheckState.Unchecked)
                self._updating = False
                QMessageBox.warning(
                    self,
                    "無法勾選",
                    f"「{entry.display_name}」目前不可執行，因此不能加入推論頁面。\n\n{entry.reason}",
                )
                return
            self.index.set_enabled(entry.entry_id, enabled)
            self.entries[entry.entry_id] = replace(entry, enabled=enabled)
        else:
            return
        self.index.save()
        self.library_changed.emit()

    def _set_all_enabled(self, enabled: bool) -> None:
        self._updating = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                run_item = self.tree.topLevelItem(index)
                for child_index in range(run_item.childCount()):
                    item = run_item.child(child_index)
                    if item.isHidden():
                        continue
                    entry = self.entries.get(str(item.data(0, _ENTRY_ROLE)))
                    if entry is None or (enabled and not entry.compatible):
                        continue
                    item.setCheckState(
                        1, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
                    )
                    self.index.set_enabled(entry.entry_id, enabled)
                    self.entries[entry.entry_id] = replace(entry, enabled=enabled)
        finally:
            self._updating = False
        self.index.save()
        self.library_changed.emit()

    def _on_selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        self.notes_edit.blockSignals(True)
        if current is None:
            self.detail_box.setPlainText("")
            self.notes_edit.setPlainText("")
            self.notes_edit.blockSignals(False)
            return
        run_folder = current.data(0, _RUN_ROLE)
        if run_folder is not None:
            self.detail_box.setPlainText(f"Run 資料夾：{run_folder}")
            self.notes_edit.setPlainText(self.index.run_notes(str(run_folder)))
            self.notes_edit.blockSignals(False)
            return
        entry = self.entries.get(str(current.data(0, _ENTRY_ROLE)))
        if entry is None:
            self.notes_edit.blockSignals(False)
            return
        lines = [
            f"檔案：{entry.path}",
            f"類型：{entry.kind_label}",
            f"架構／模型：{entry.architecture or entry.model_name or '—'}",
            f"Run：{entry.run_label}",
        ]
        if entry.experiment:
            lines.append(f"Experiment：{entry.experiment}")
        if entry.target_task:
            lines.append(f"Target Task：{entry.target_task}")
        if entry.rank is not None:
            lines.append(f"該 Target Task 內排名：第 {entry.rank} 名")
        if entry.evidence_stage:
            lines.append(f"證據階段：{entry.evidence_stage}")
        if entry.trained_at:
            lines.append(f"儲存時間：{entry.trained_at}")
        if entry.metrics:
            metric_text = "、".join(
                f"{key}={value:.6g}" for key, value in sorted(entry.metrics.items())
            )
            lines.append(f"指標：{metric_text}")
        if entry.feature_names:
            lines.append(f"輸入（{len(entry.feature_names)}）：{'、'.join(entry.feature_names)}")
        if entry.target_names:
            lines.append(f"輸出（{len(entry.target_names)}）：{'、'.join(entry.target_names)}")
        lines.append(f"狀態：{entry.reason}")
        self.detail_box.setPlainText("\n".join(lines))
        self.notes_edit.setPlainText(entry.notes)
        self.notes_edit.blockSignals(False)

    def _on_notes_changed(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        text = self.notes_edit.toPlainText()
        run_folder = item.data(0, _RUN_ROLE)
        if run_folder is not None:
            self.index.set_run_notes(str(run_folder), text)
        else:
            entry = self.entries.get(str(item.data(0, _ENTRY_ROLE)))
            if entry is None:
                return
            self.index.set_notes(entry.entry_id, text)
            self.entries[entry.entry_id] = replace(entry, notes=text)
        self.index.save()

    # ------------------------------------------------------------ filtering
    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        only_usable = self.only_usable_check.isChecked()
        for index in range(self.tree.topLevelItemCount()):
            run_item = self.tree.topLevelItem(index)
            visible_children = 0
            for child_index in range(run_item.childCount()):
                item = run_item.child(child_index)
                entry = self.entries.get(str(item.data(0, _ENTRY_ROLE)))
                if entry is None:
                    continue
                haystack = " ".join(
                    (
                        entry.display_name,
                        entry.run_label,
                        entry.model_name,
                        entry.architecture,
                        entry.target_task,
                        entry.reason,
                        " ".join(entry.target_names),
                    )
                ).lower()
                hidden = bool(needle) and needle not in haystack
                if only_usable and not entry.compatible:
                    hidden = True
                item.setHidden(hidden)
                visible_children += 0 if hidden else 1
            run_item.setHidden(visible_children == 0)

    # -------------------------------------------------------------- actions
    def _show_context_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        open_action = menu.addAction("在檔案總管開啟位置")
        remove_action = None
        entry_id = item.data(0, _ENTRY_ROLE)
        if entry_id is not None:
            remove_action = menu.addAction("從模型庫移除（刪除檔案）")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(position))
        if chosen is None:
            return
        if chosen is open_action:
            target = (
                Path(str(item.data(0, _RUN_ROLE)))
                if entry_id is None
                else self.entries[str(entry_id)].path.parent
            )
            self._open_folder(target)
        elif remove_action is not None and chosen is remove_action:
            self._remove_entry(str(entry_id))

    @staticmethod
    def _open_folder(folder: Path) -> None:
        import os

        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            pass

    def _remove_entry(self, entry_id: str) -> None:
        entry = self.entries.get(entry_id)
        if entry is None:
            return
        answer = QMessageBox.question(
            self,
            "刪除模型檔",
            f"要永久刪除這個檔案嗎？\n\n{entry.path}\n\n（同名的 .meta.json / .md 說明檔也會一併刪除）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for candidate in (
            entry.path,
            entry.path.with_suffix(".meta.json"),
            entry.path.with_name(entry.path.stem + ".meta.json"),
            entry.path.with_suffix(".md"),
        ):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                QMessageBox.warning(self, "刪除失敗", f"{candidate}\n{exc}")
                return
        self.index.forget(entry_id)
        self.index.save()
        self.refresh()

    def import_models(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in MODEL_SUFFIXES)
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "選擇要匯入模型庫的檔案", str(MODEL_LIBRARY_ROOT), f"模型檔 ({patterns})"
        )
        if not paths:
            return
        destination = MODEL_LIBRARY_ROOT / "imported"
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for raw in paths:
            source = Path(raw)
            target = destination / source.name
            if target.exists():
                answer = QMessageBox.question(
                    self,
                    "檔案已存在",
                    f"{target.name} 已經在模型庫中，要覆蓋嗎？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    continue
            try:
                shutil.copy2(source, target)
                for extra in (".md", ".meta.json"):
                    sidecar = source.with_suffix(extra)
                    if sidecar.exists():
                        shutil.copy2(sidecar, destination / sidecar.name)
                copied += 1
            except OSError as exc:
                QMessageBox.warning(self, "匯入失敗", f"{source}\n{exc}")
        if copied:
            self.refresh()

    # ----------------------------------------------------------- public API
    def usable_entries(self) -> list[ModelEntry]:
        """Models the user ticked and that actually load, for the inference tabs."""

        return sorted(
            (entry for entry in self.entries.values() if entry.usable),
            key=lambda entry: (entry.run_label, entry.rank or 99, entry.display_name),
        )

    def entry(self, entry_id: str) -> ModelEntry | None:
        return self.entries.get(entry_id)

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)


def _brush(colour: str):
    from PyQt6.QtGui import QBrush, QColor

    return QBrush(QColor(colour))
