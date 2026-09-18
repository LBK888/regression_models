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

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QTabWidget

from project_paths import ensure_folders
from regression_core import DEVICE, EXECUTION_MODE_LABEL
from regression_v4.ui import TrainingPage

from .batch_page import BatchPage
from .inference_page import InferencePage
from .library_page import LibraryPage
from .theme import C, apply_theme


APP_TITLE = "Regression Models — 訓練、模型庫與推論"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_folders()
        self.setWindowTitle(f"{APP_TITLE}　[{EXECUTION_MODE_LABEL}]")
        self.resize(1560, 960)

        self.training_page = TrainingPage()
        self.library_page = LibraryPage()
        self.inference_page = InferencePage()
        self.batch_page = BatchPage()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.training_page, "① 訓練與超參數試探")
        self.tabs.addTab(self.library_page, "② 模型庫")
        self.tabs.addTab(self.inference_page, "③ 單筆推論")
        self.tabs.addTab(self.batch_page, "④ 批次推論與驗證")
        self.setCentralWidget(self.tabs)

        status = QStatusBar()
        device_label = QLabel(
            f"運算裝置：{DEVICE.type.upper()}"
            + (f"（{__import__('torch').cuda.get_device_name(0)}）" if DEVICE.type == "cuda" else "")
        )
        device_label.setStyleSheet(f"color: {C['muted']}; padding: 0 8px;")
        status.addPermanentWidget(device_label)
        self.setStatusBar(status)

        self.training_page.run_completed.connect(self._on_run_completed)
        self.library_page.library_changed.connect(self._publish_models)
        self._publish_models()

    def _on_run_completed(self, run_folder: str) -> None:
        """A finished Benchmark Run just wrote new bundles; pick them up."""

        self.statusBar().showMessage(f"訓練完成：{run_folder}　正在更新模型庫…", 8000)
        self.library_page.refresh()

    def _publish_models(self) -> None:
        entries = self.library_page.usable_entries()
        self.inference_page.set_entries(entries)
        self.batch_page.set_entries(entries)
        self.statusBar().showMessage(f"可用於推論的模型：{len(entries)} 個", 5000)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.training_page.can_close():
            event.ignore()
            return
        self.library_page.stop()
        self.inference_page.stop()
        self.batch_page.stop()
        event.accept()


def launch(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv if argv is not None else sys.argv))
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
