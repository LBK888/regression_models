# -*- coding: utf-8 -*-
"""Application layer: the merged UI, the model library and the inference stack.

``regression_v4`` owns training and evaluation; ``regression_core`` owns data
loading and the shared architectures. This package owns everything that turns a
saved model back into predictions, and the single window that ties the two
halves together.
"""

from __future__ import annotations

__all__ = ["launch"]

__version__ = "1.0.0"


def launch(argv=None) -> int:
    """Start the merged application."""

    from .main_window import launch as _launch

    return _launch(argv)
