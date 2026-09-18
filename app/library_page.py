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

from i18n import tr
from project_paths import MODEL_LIBRARY_ROOT, ensure_folders

from .adapters import MODEL_SUFFIXES
from .model_library import LibraryIndex, ModelEntry, build_entry, discover_model_files, probe_entry
from .theme import C


def column_labels() -> tuple[str, ...]:
    """Header labels, resolved on each build so a language switch re-renders them."""

    return (
        tr("Name / Run"),
        tr("Use"),
        tr("Model"),
        "R²",
        "MAPE %",
        tr("Inputs"),
        tr("Outputs"),
        tr("Kind"),
        tr("Status"),
    )


COLUMN_COUNT = 9

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
        self.title_label = QLabel(tr("Model library"))
        self.title_label.setStyleSheet(
            f"color: {C['accent2']}; font-size: 18px; font-weight: bold;"
        )
        self.status_label = QLabel(tr("Not scanned yet"))
        self.status_label.setStyleSheet(f"color: {C['muted']};")
        header.addWidget(self.title_label)
        header.addWidget(self.status_label, 1)

        self.rescan_button = QPushButton(tr("🔄 Rescan"))
        self.rescan_button.setObjectName("secondaryBtn")
        self.rescan_button.clicked.connect(self.refresh)
        self.import_button = QPushButton(tr("➕ Import model files"))
        self.import_button.setObjectName("secondaryBtn")
        self.import_button.clicked.connect(self.import_models)
        header.addWidget(self.import_button)
        header.addWidget(self.rescan_button)
        layout.addLayout(header)

        self.hint_label = QLabel(
            tr(
                "Name your runs and models here, and tick the ones you want to see on "
                "the inference tabs. Third-party and legacy models are listed too; "
                "anything that cannot run says why in the Status column."
            )
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        layout.addWidget(self.hint_label)

        tools = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            tr("Filter by run, model name, target or status…")
        )
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.only_usable_check = QCheckBox(tr("Show runnable models only"))
        self.only_usable_check.toggled.connect(self._apply_filter)
        self.enable_all_button = QPushButton(tr("Tick all"))
        self.enable_all_button.setObjectName("secondaryBtn")
        self.enable_all_button.clicked.connect(lambda: self._set_all_enabled(True))
        self.disable_all_button = QPushButton(tr("Untick all"))
        self.disable_all_button.setObjectName("secondaryBtn")
        self.disable_all_button.clicked.connect(lambda: self._set_all_enabled(False))
        tools.addWidget(self.filter_edit, 1)
        tools.addWidget(self.only_usable_check)
        tools.addWidget(self.enable_all_button)
        tools.addWidget(self.disable_all_button)
        layout.addLayout(tools)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(COLUMN_COUNT)
        self.tree.setHeaderLabels(column_labels())
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
        for column in range(1, COLUMN_COUNT):
            self.tree.setColumnWidth(column, 110)
        splitter.addWidget(self.tree)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        self.detail_caption = QLabel(
            tr("Details and notes for the selected item (notes are saved automatically)")
        )
        detail_layout.addWidget(self.detail_caption)
        self.detail_box = QPlainTextEdit()
        self.detail_box.setReadOnly(True)
        self.detail_box.setMaximumHeight(150)
        detail_layout.addWidget(self.detail_box)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            tr("Describe what this model or run is for…")
        )
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.textChanged.connect(self._on_notes_changed)
        detail_layout.addWidget(self.notes_edit)
        splitter.addWidget(detail)
        splitter.setSizes([560, 220])
        layout.addWidget(splitter, 1)

    def retranslate(self) -> None:
        """Re-apply every static string after a language change."""

        self.title_label.setText(tr("Model library"))
        self.rescan_button.setText(tr("🔄 Rescan"))
        self.import_button.setText(tr("➕ Import model files"))
        self.hint_label.setText(
            tr(
                "Name your runs and models here, and tick the ones you want to see on "
                "the inference tabs. Third-party and legacy models are listed too; "
                "anything that cannot run says why in the Status column."
            )
        )
        self.filter_edit.setPlaceholderText(tr("Filter by run, model name, target or status…"))
        self.only_usable_check.setText(tr("Show runnable models only"))
        self.enable_all_button.setText(tr("Tick all"))
        self.disable_all_button.setText(tr("Untick all"))
        self.detail_caption.setText(
            tr("Details and notes for the selected item (notes are saved automatically)")
        )
        self.notes_edit.setPlaceholderText(tr("Describe what this model or run is for…"))
        self.tree.setHeaderLabels(column_labels())
        # Every row's Kind and Status cell comes from a probe, so a rescan is
        # what re-renders the table itself in the new language.
        self.refresh()

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
        self.status_label.setText(tr("Scanning…"))
        self.worker = ScanWorker(self.index)
        self.worker.entry_found.connect(self._add_entry)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_scan.connect(self._on_scan_finished)
        self.worker.start()

    def _on_progress(self, current: int, total: int, name: str) -> None:
        self.status_label.setText(
            tr("Scanning {current}/{total}: {name}", current=current, total=total, name=name)
        )

    def _on_scan_finished(self, total: int) -> None:
        self.rescan_button.setEnabled(True)
        self.index.prune(self.entries.keys())
        self.index.save()
        usable = sum(1 for entry in self.entries.values() if entry.compatible)
        enabled = sum(1 for entry in self.entries.values() if entry.usable)
        self.status_label.setText(
            tr(
                "{total} model file(s); {usable} runnable; {enabled} ticked for inference",
                total=total,
                usable=usable,
                enabled=enabled,
            )
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
                    tr("Cannot be ticked"),
                    tr(
                        "“{name}” cannot run, so it cannot be added to the "
                        "inference tabs.\n\n{reason}",
                        name=entry.display_name,
                        reason=entry.reason,
                    ),
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
            self.detail_box.setPlainText(tr("Run folder: {path}", path=str(run_folder)))
            self.notes_edit.setPlainText(self.index.run_notes(str(run_folder)))
            self.notes_edit.blockSignals(False)
            return
        entry = self.entries.get(str(current.data(0, _ENTRY_ROLE)))
        if entry is None:
            self.notes_edit.blockSignals(False)
            return
        lines = [
            tr("File: {path}", path=str(entry.path)),
            tr("Kind: {kind}", kind=entry.kind_label),
            tr(
                "Architecture / model: {name}",
                name=entry.architecture or entry.model_name or "—",
            ),
            tr("Run: {name}", name=entry.run_label),
        ]
        if entry.experiment:
            lines.append(tr("Experiment: {name}", name=entry.experiment))
        if entry.target_task:
            lines.append(tr("Target Task: {name}", name=entry.target_task))
        if entry.rank is not None:
            lines.append(tr("Rank inside this Target Task: {rank}", rank=entry.rank))
        if entry.evidence_stage:
            lines.append(tr("Evidence stage: {stage}", stage=entry.evidence_stage))
        if entry.trained_at:
            lines.append(tr("Saved at: {timestamp}", timestamp=entry.trained_at))
        if entry.metrics:
            metric_text = ", ".join(
                f"{key}={value:.6g}" for key, value in sorted(entry.metrics.items())
            )
            lines.append(tr("Metrics: {metrics}", metrics=metric_text))
        if entry.feature_names:
            lines.append(
                tr(
                    "Inputs ({count}): {names}",
                    count=len(entry.feature_names),
                    names=", ".join(entry.feature_names),
                )
            )
        if entry.target_names:
            lines.append(
                tr(
                    "Outputs ({count}): {names}",
                    count=len(entry.target_names),
                    names=", ".join(entry.target_names),
                )
            )
        lines.append(tr("Status: {status}", status=entry.reason))
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
        open_action = menu.addAction(tr("Open the containing folder"))
        remove_action = None
        entry_id = item.data(0, _ENTRY_ROLE)
        if entry_id is not None:
            remove_action = menu.addAction(tr("Remove from the library (deletes the file)"))
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
            tr("Delete model file"),
            tr(
                "Permanently delete this file?\n\n{path}\n\nAny matching .meta.json "
                "and .md sidecar is deleted with it.",
                path=str(entry.path),
            ),
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
                QMessageBox.warning(self, tr("Delete failed"), f"{candidate}\n{exc}")
                return
        self.index.forget(entry_id)
        self.index.save()
        self.refresh()

    def import_models(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in MODEL_SUFFIXES)
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            tr("Choose model files to import"),
            str(MODEL_LIBRARY_ROOT),
            tr("Model files ({patterns})", patterns=patterns),
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
                    tr("File already exists"),
                    tr("{name} is already in the library. Overwrite it?", name=target.name),
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
                QMessageBox.warning(self, tr("Import failed"), f"{source}\n{exc}")
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
