# -*- coding: utf-8 -*-
"""The single window that hosts training, the model library and inference.

The four tabs are one workflow read left to right: train models, decide which
of them are worth keeping visible, predict one sample, then predict and
validate a whole table. The library tab is the hinge — it is the only place
that decides which models the two inference tabs can see.
"""

from __future__ import annotations

import sys
from typing import Sequence

import torch

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

import i18n
from i18n import tr
from project_paths import ensure_folders
from regression_core import DEVICE, EXECUTION_MODE_LABEL
from regression_v4.ui import TrainingPage

from .batch_page import BatchPage
from .inference_page import InferencePage
from .library_page import LibraryPage
from .theme import C, apply_theme


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_folders()

        self.training_page = TrainingPage()
        self.library_page = LibraryPage()
        self.inference_page = InferencePage()
        self.batch_page = BatchPage()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.training_page, "")
        self.tabs.addTab(self.library_page, "")
        self.tabs.addTab(self.inference_page, "")
        self.tabs.addTab(self.batch_page, "")
        self.setCentralWidget(self.tabs)

        status = QStatusBar()
        self.language_caption = QLabel()
        self.language_caption.setStyleSheet(f"color: {C['muted']}; padding: 0 4px;")
        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(130)
        for code, name in i18n.available_languages():
            self.language_combo.addItem(name, code)
        index = self.language_combo.findData(i18n.current_language())
        self.language_combo.setCurrentIndex(max(0, index))
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        self.device_label = QLabel()
        self.device_label.setStyleSheet(f"color: {C['muted']}; padding: 0 8px;")
        status.addPermanentWidget(self.language_caption)
        status.addPermanentWidget(self.language_combo)
        status.addPermanentWidget(self.device_label)
        self.setStatusBar(status)

        self.resize(1560, 960)
        self.retranslate()

        self.training_page.run_completed.connect(self._on_run_completed)
        self.library_page.library_changed.connect(self._publish_models)
        self._publish_models()

    # ----------------------------------------------------------------- i18n
    def retranslate(self) -> None:
        """Re-apply the window's own strings. Pages retranslate themselves."""

        self.setWindowTitle(
            tr("Regression Models — training, model library and inference")
            + f"　[{tr(EXECUTION_MODE_LABEL)}]"
        )
        self.tabs.setTabText(0, tr("① Training and hyperparameter search"))
        self.tabs.setTabText(1, tr("② Model library"))
        self.tabs.setTabText(2, tr("③ Single-sample inference"))
        self.tabs.setTabText(3, tr("④ Batch inference and validation"))
        self.language_caption.setText(tr("Language"))
        device = tr("Compute device: {device}", device=DEVICE.type.upper())
        if DEVICE.type == "cuda":
            device += f" ({torch.cuda.get_device_name(0)})"
        self.device_label.setText(device)

    def _on_language_selected(self, index: int) -> None:
        language = self.language_combo.itemData(index)
        if not language or language == i18n.current_language():
            return
        if self.training_page.has_running_worker() or (
            self.batch_page.worker is not None and self.batch_page.worker.isRunning()
        ):
            QMessageBox.warning(
                self,
                tr("A run is in progress"),
                tr("Wait for the current run to finish before switching language."),
            )
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(
                self.language_combo.findData(i18n.current_language())
            )
            self.language_combo.blockSignals(False)
            return

        previous = self.tabs.currentIndex()
        i18n.set_language(language)
        self.retranslate()
        # The training tab is rebuilt rather than relabelled: its settings tab
        # builds dozens of labels and tooltips inline. A loaded dataset is
        # reloaded from its path as part of that rebuild.
        self.training_page.retranslate()
        self.library_page.retranslate()
        self.inference_page.retranslate()
        self.batch_page.retranslate()
        self.tabs.setCurrentIndex(previous)
        self.statusBar().showMessage(
            tr("Interface language: {name}", name=i18n.language_name(language)), 5000
        )

    # ------------------------------------------------------------- plumbing
    def _on_run_completed(self, run_folder: str) -> None:
        """A finished Benchmark Run just wrote new bundles; pick them up."""

        self.statusBar().showMessage(
            tr("Training finished: {path} — refreshing the model library…", path=run_folder),
            8000,
        )
        self.library_page.refresh()

    def _publish_models(self) -> None:
        entries = self.library_page.usable_entries()
        self.inference_page.set_entries(entries)
        self.batch_page.set_entries(entries)
        self.statusBar().showMessage(
            tr("Models available for inference: {count}", count=len(entries)), 5000
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.training_page.can_close():
            event.ignore()
            return
        self.library_page.stop()
        self.inference_page.stop()
        self.batch_page.stop()
        event.accept()


def launch(argv: Sequence[str] | None = None) -> int:
    i18n.initialize()
    app = QApplication(list(argv if argv is not None else sys.argv))
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
