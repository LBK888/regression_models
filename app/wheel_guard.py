# -*- coding: utf-8 -*-
"""Stop the mouse wheel from changing settings while the user scrolls a page.

Qt lets the wheel step a spin box, switch a combo box or move a slider whenever
the pointer happens to be over it. On a long settings page that silently
rewrites hyper-parameters mid-scroll. This application-wide filter swallows the
wheel for those widgets and lets the event continue to the parent, so the page
scrolls instead. Values still change with the keyboard, the arrow buttons and
the drop-down list.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
)


class WheelGuard(QObject):
    """Application event filter: value widgets ignore the wheel, pages scroll."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and _is_value_widget(watched):
            # Returning True without accepting lets QApplication offer the same
            # wheel event to the parent widget — normally a scroll area.
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def _is_value_widget(widget: QObject) -> bool:
    if isinstance(widget, QScrollBar):
        return False
    return isinstance(widget, (QAbstractSpinBox, QComboBox, QAbstractSlider))


_GUARD: WheelGuard | None = None


def install_wheel_guard(app: QApplication) -> WheelGuard:
    """Install the filter once per application; safe to call repeatedly."""

    global _GUARD
    if _GUARD is None:
        _GUARD = WheelGuard(app)
        app.installEventFilter(_GUARD)
    return _GUARD
