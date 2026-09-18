# -*- coding: utf-8 -*-
"""Single-sample inference, with the form built from the selected model.

Nothing about the layout is fixed: the number of input fields, their names, the
number of result cards and their names all come from the loaded model. A
``regression_v4`` bundle carries the Experiment's input and output names; a
legacy package carries them in the pickle or in its ``.md`` note. Switching
models rebuilds the form.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from .adapters import Predictor, load_predictor
from .model_library import ModelEntry
from .theme import C


class FeatureRow(QWidget):
    """One named numeric input."""

    def __init__(self, feature_name: str, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.feature_name = feature_name
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 3, 4, 3)
        row.setSpacing(12)

        number = QLabel(f"{index + 1:02d}")
        number.setFixedWidth(28)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setStyleSheet(f"color: {C['muted']}; font-size: 11px; font-family: monospace;")

        name = QLabel(feature_name)
        name.setMinimumWidth(180)
        name.setMaximumWidth(300)
        name.setToolTip(feature_name)
        name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        name.setStyleSheet(f"color: {C['text']}; font-size: 13px;")

        self.field = QLineEdit()
        self.field.setPlaceholderText("輸入數值…")
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.field.setValidator(validator)

        row.addWidget(number)
        row.addWidget(name)
        row.addWidget(self.field, 1)

    def value(self) -> float:
        text = self.field.text().strip()
        if not text:
            raise ValueError(f"「{self.feature_name}」尚未填寫")
        try:
            return float(text.replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"「{self.feature_name}」格式不正確：{text!r}") from exc

    def set_value(self, value: Any) -> None:
        self.field.setText("" if value is None else str(value))

    def clear(self) -> None:
        self.field.clear()


class ResultCard(QFrame):
    """One predicted target value."""

    def __init__(self, target_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.target_name = target_name
        self.setStyleSheet(
            f"ResultCard {{ background-color: {C['panel']}; border: 1px solid {C['border']}; "
            f"border-radius: 12px; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(4)

        caption = QLabel("預測目標")
        caption.setStyleSheet(f"color: {C['muted']}; font-size: 11px; font-weight: bold;")
        name = QLabel(target_name)
        name.setWordWrap(True)
        name.setStyleSheet(f"color: {C['text']}; font-size: 15px; font-weight: bold;")

        self.value_label = QLabel("—")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._style_value(C["muted"])

        layout.addWidget(caption)
        layout.addWidget(name)
        layout.addWidget(self.value_label)

    def _style_value(self, colour: str) -> None:
        self.value_label.setStyleSheet(
            f"color: {colour}; font-size: 34px; font-weight: bold; padding: 6px 0 2px 0;"
        )

    def show_value(self, value: float) -> None:
        self.value_label.setText(f"{value:.6g}")
        self._style_value(C["success"])

    def reset(self) -> None:
        self.value_label.setText("—")
        self._style_value(C["muted"])


class LoadWorker(QThread):
    """Load a model off the UI thread; a first CUDA touch can take seconds."""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            self.loaded.emit(load_predictor(self.path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class InferencePage(QWidget):
    """Pick a model, fill in its inputs, read its outputs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entries: list[ModelEntry] = []
        self.predictor: Predictor | None = None
        self.current_entry: ModelEntry | None = None
        self.feature_rows: list[FeatureRow] = []
        self.result_cards: list[ResultCard] = []
        self.loader: LoadWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        selector = QGroupBox("模型選擇")
        selector_layout = QVBoxLayout(selector)
        row = QHBoxLayout()
        label = QLabel("使用模型：")
        label.setStyleSheet(f"color: {C['muted']}; font-weight: bold;")
        label.setFixedWidth(72)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(label)
        row.addWidget(self.model_combo, 1)
        selector_layout.addLayout(row)

        self.status_label = QLabel("請先到「模型庫」勾選要使用的模型。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {C['muted']}; font-size: 12px; padding: 2px 0;")
        selector_layout.addWidget(self.status_label)

        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(f"color: {C['warning']}; font-size: 12px;")
        selector_layout.addWidget(self.metrics_label)
        layout.addWidget(selector)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        input_box = QGroupBox("輸入特徵值")
        input_layout = QVBoxLayout(input_box)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(260)
        self.inputs_widget = QWidget()
        self.inputs_widget.setStyleSheet("background: transparent;")
        self.inputs_layout = QVBoxLayout(self.inputs_widget)
        self.inputs_layout.setSpacing(3)
        self.inputs_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.inputs_widget)
        input_layout.addWidget(self.scroll, 1)

        buttons = QHBoxLayout()
        self.predict_button = QPushButton("🔮  預測")
        self.predict_button.setEnabled(False)
        self.predict_button.setMinimumHeight(42)
        self.predict_button.clicked.connect(self.run_inference)
        clear_button = QPushButton("🗑 清除")
        clear_button.setObjectName("secondaryBtn")
        clear_button.setFixedWidth(96)
        clear_button.clicked.connect(self.clear_inputs)
        buttons.addWidget(self.predict_button, 1)
        buttons.addWidget(clear_button)
        input_layout.addLayout(buttons)
        splitter.addWidget(input_box)

        result_box = QGroupBox("推論結果")
        self.result_layout = QVBoxLayout(result_box)
        self.result_layout.setSpacing(10)
        self.result_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result_hint = QLabel("等待推論…")
        self.result_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_hint.setStyleSheet(f"color: {C['muted']}; font-size: 14px; padding: 50px 16px;")
        self.result_layout.addWidget(self.result_hint)
        splitter.addWidget(result_box)
        splitter.setSizes([700, 420])
        layout.addWidget(splitter, 1)

    # ----------------------------------------------------------- population
    def set_entries(self, entries: list[ModelEntry]) -> None:
        """Refill the model list from the library's ticked, runnable models."""

        previous = self.model_combo.currentData()
        self.entries = list(entries)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("— 請選擇模型 —", None)
        current_run = ""
        for entry in self.entries:
            if entry.run_label != current_run:
                current_run = entry.run_label
                self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItem(f"{entry.run_label} / {entry.summary_line()}", entry.entry_id)
        index = self.model_combo.findData(previous) if previous else -1
        self.model_combo.setCurrentIndex(max(0, index))
        self.model_combo.blockSignals(False)

        if not self.entries:
            self.status_label.setText(
                "目前沒有已勾選且可執行的模型。請到「模型庫」分頁勾選，或先完成一次訓練。"
            )
            self._clear_form()
        elif index <= 0:
            self.status_label.setText(f"可用模型 {len(self.entries)} 個，請選擇一個開始推論。")
        else:
            self._on_model_changed(self.model_combo.currentIndex())

    def _on_model_changed(self, index: int) -> None:
        entry_id = self.model_combo.itemData(index)
        if entry_id is None:
            self.current_entry = None
            self.predictor = None
            self._clear_form()
            self.predict_button.setEnabled(False)
            self.metrics_label.setText("")
            return
        entry = next((item for item in self.entries if item.entry_id == entry_id), None)
        if entry is None:
            return
        self.current_entry = entry
        self.predict_button.setEnabled(False)
        self._set_status("⏳ 載入模型中…", C["muted"])
        self.loader = LoadWorker(str(entry.path))
        self.loader.loaded.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def _on_loaded(self, predictor: Predictor) -> None:
        self.predictor = predictor
        entry = self.current_entry
        self._rebuild_form(predictor)
        self.predict_button.setEnabled(True)
        notes = f"（{'; '.join(predictor.notes)}）" if predictor.notes else ""
        self._set_status(
            f"✅ 已載入：{predictor.architecture or predictor.kind}　"
            f"輸入 {len(predictor.feature_names)}　輸出 {len(predictor.target_names)}　"
            f"裝置 {predictor.device}{notes}",
            C["success"],
        )
        if entry and entry.metrics:
            parts = []
            if entry.reported_r2 is not None:
                parts.append(f"R²={entry.reported_r2:.4f}")
            if entry.mape_percent is not None:
                parts.append(f"MAPE={entry.mape_percent:.2f}%")
            for key in ("mae", "rmse", "nmae"):
                if key in entry.metrics:
                    parts.append(f"{key.upper()}={entry.metrics[key]:.6g}")
            self.metrics_label.setText(
                "訓練時記錄的評估指標： " + "　".join(parts)
                + "　（來自該模型的評估階段，不是本次輸入的準確度）"
            )
        else:
            self.metrics_label.setText("這個模型沒有記錄評估指標。")

    def _on_load_failed(self, message: str) -> None:
        self.predictor = None
        self._clear_form()
        self.predict_button.setEnabled(False)
        self._set_status(f"❌ 載入失敗：{message}", C["error"])

    # ----------------------------------------------------------------- form
    def _rebuild_form(self, predictor: Predictor) -> None:
        self._clear_inputs()
        for index, name in enumerate(predictor.feature_names):
            row = FeatureRow(name, index)
            self.feature_rows.append(row)
            self.inputs_layout.addWidget(row)

        self._clear_results()
        for name in predictor.target_names:
            card = ResultCard(name)
            self.result_cards.append(card)
            self.result_layout.addWidget(card)
        self.result_layout.addStretch(1)

    def _clear_inputs(self) -> None:
        self.feature_rows.clear()
        while self.inputs_layout.count():
            item = self.inputs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_results(self) -> None:
        self.result_cards.clear()
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.result_hint:
                widget.deleteLater()

    def _clear_form(self) -> None:
        self._clear_inputs()
        self._clear_results()
        self.result_layout.addWidget(self.result_hint)
        self.result_hint.show()

    def clear_inputs(self) -> None:
        for row in self.feature_rows:
            row.clear()
        for card in self.result_cards:
            card.reset()

    # ------------------------------------------------------------ inference
    def run_inference(self) -> None:
        if self.predictor is None:
            return
        try:
            values = [row.value() for row in self.feature_rows]
        except ValueError as exc:
            QMessageBox.warning(self, "輸入錯誤", str(exc))
            return
        try:
            prediction = self.predictor.predict(np.asarray([values], dtype=float))
        except Exception as exc:
            QMessageBox.critical(self, "推論錯誤", f"{type(exc).__name__}: {exc}")
            return
        for index, card in enumerate(self.result_cards):
            card.show_value(float(prediction[0, index]))

    def _set_status(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {colour}; font-size: 12px; padding: 2px 0;")

    def stop(self) -> None:
        if self.loader and self.loader.isRunning():
            self.loader.wait(3000)
