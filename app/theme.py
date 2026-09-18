# -*- coding: utf-8 -*-
"""Shared dark theme, carried over from the AI_Sensor inference UI.

The whole merged application uses one palette so the training, library,
inference and batch tabs do not look like three different programs.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


C = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "panel": "#22263a",
    "border": "#2e3350",
    "accent": "#5b6ef5",
    "accent2": "#7c8cff",
    "text": "#d4d8f0",
    "muted": "#666e99",
    "success": "#4fc97e",
    "warning": "#f5a623",
    "error": "#f56565",
    "white": "#ffffff",
}

APP_STYLE = f"""
QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Microsoft JhengHei UI', 'Segoe UI', 'PingFang TC', Arial, sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: bold;
    color: {C['accent2']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: {C['accent2']};
    background-color: {C['surface']};
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QTextEdit {{
    background-color: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 7px;
    padding: 5px 10px;
    color: {C['text']};
    min-height: 26px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {C['muted']};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {C['accent2']};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {C['panel']};
    border: 1px solid {C['border']};
    selection-background-color: {C['accent']};
    color: {C['text']};
    padding: 4px;
}}

QPushButton {{
    background-color: {C['accent']};
    color: {C['white']};
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: bold;
    min-height: 32px;
}}
QPushButton:hover {{ background-color: {C['accent2']}; }}
QPushButton:pressed {{ background-color: #4a5bd4; }}
QPushButton:disabled {{ background-color: {C['border']}; color: {C['muted']}; }}
QPushButton#secondaryBtn {{
    background-color: {C['panel']};
    color: {C['text']};
    border: 1px solid {C['border']};
}}
QPushButton#secondaryBtn:hover {{ border-color: {C['muted']}; background-color: #2a2e45; }}

QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {C['border']};
    border-radius: 4px;
    background: {C['panel']};
}}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QCheckBox::indicator:disabled {{ background: {C['surface']}; border-color: {C['surface']}; }}

QTabWidget::pane {{ border: 1px solid {C['border']}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: {C['surface']};
    color: {C['muted']};
    border: 1px solid {C['border']};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 18px;
    margin-right: 2px;
    font-weight: bold;
}}
QTabBar::tab:selected {{ background: {C['panel']}; color: {C['accent2']}; }}
QTabBar::tab:hover:!selected {{ color: {C['text']}; }}

QTableView, QTreeWidget, QTreeView, QTableWidget {{
    background-color: {C['surface']};
    alternate-background-color: #1e2130;
    border: 1px solid {C['border']};
    border-radius: 8px;
    gridline-color: {C['border']};
    selection-background-color: {C['accent']};
    selection-color: {C['white']};
}}
QHeaderView::section {{
    background-color: {C['panel']};
    color: {C['accent2']};
    border: none;
    border-right: 1px solid {C['border']};
    border-bottom: 1px solid {C['border']};
    padding: 6px 8px;
    font-weight: bold;
}}

QProgressBar {{
    background-color: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 7px;
    height: 18px;
    text-align: center;
    color: {C['text']};
}}
QProgressBar::chunk {{ background-color: {C['accent']}; border-radius: 6px; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: {C['surface']}; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {C['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {C['surface']}; height: 8px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: {C['border']}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QSplitter::handle {{ background: {C['border']}; }}
QToolTip {{
    background-color: {C['panel']};
    color: {C['text']};
    border: 1px solid {C['accent']};
    padding: 6px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the Fusion style, the dark palette and the shared stylesheet."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(C["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(C["panel"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(C["surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(C["panel"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(C["panel"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(C["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(C["white"]))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE)
