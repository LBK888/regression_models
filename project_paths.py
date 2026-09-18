# -*- coding: utf-8 -*-
"""Single source of truth for the folders the application reads and writes.

Everything lives beside the program so the whole folder stays portable: copy
``regression_models`` to another machine, run ``install.bat``, and the model
library and previous run outputs come along unchanged.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

#: Benchmark Runs. Each run gets its own ``run_<timestamp>`` child folder that
#: holds the report, the audit CSVs and a ``models/`` folder of saved bundles.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs"

#: Standalone models: the two legacy v2.2 sensor models plus anything the user
#: drops in, including third-party regressors.
MODEL_LIBRARY_ROOT = PROJECT_ROOT / "models"

#: User-owned naming, enable flags and notes for library entries.
LIBRARY_INDEX_PATH = MODEL_LIBRARY_ROOT / "library.json"

#: Batch inference and validation reports.
REPORTS_ROOT = PROJECT_ROOT / "reports"

#: User preferences that are not tied to one model, such as the UI language.
SETTINGS_PATH = PROJECT_ROOT / "settings.json"


def ensure_folders() -> None:
    for folder in (DEFAULT_OUTPUT_ROOT, MODEL_LIBRARY_ROOT, REPORTS_ROOT):
        folder.mkdir(parents=True, exist_ok=True)


def model_search_roots() -> tuple[Path, ...]:
    """Folders scanned for models: the library plus every Benchmark Run."""

    return (MODEL_LIBRARY_ROOT, DEFAULT_OUTPUT_ROOT)
