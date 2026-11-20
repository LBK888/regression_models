# -*- coding: utf-8 -*-
"""Single source of truth for the folders the application reads and writes.

Everything lives beside the program so the whole folder stays portable: copy
``regression_models`` to another machine, run ``install.bat``, and the model
library and previous run outputs come along unchanged.

The packaged ``RegressionModels.exe`` follows the same rule: its writable
folders sit next to the exe, while read-only files bundled into the exe (the
seed models) live in PyInstaller's extraction folder, ``RESOURCE_ROOT``.
"""

from __future__ import annotations

import sys
from pathlib import Path


#: True inside the PyInstaller build.
FROZEN = bool(getattr(sys, "frozen", False))

#: The folder the user sees: the exe's folder when packaged, else the source tree.
PROJECT_ROOT = (
    Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
)

#: Read-only files shipped inside the build (``_internal`` when packaged).
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

#: Benchmark Runs. Each run gets its own ``run_<timestamp>`` child folder that
#: holds the report, the audit CSVs and a ``models/`` folder of saved bundles.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs"

#: Standalone models: the two legacy v2.2 sensor models plus anything the user
#: drops in, including third-party regressors.
MODEL_LIBRARY_ROOT = PROJECT_ROOT / "models"

#: Example models bundled into the exe and copied to ``models/`` on first start.
SEED_MODELS_ROOT = RESOURCE_ROOT / "seed" / "models"

#: User-owned naming, enable flags and notes for library entries.
LIBRARY_INDEX_PATH = MODEL_LIBRARY_ROOT / "library.json"

#: Batch inference and validation reports.
REPORTS_ROOT = PROJECT_ROOT / "reports"

#: Downloaded pretrained weights (TabPFN v2). Kept beside the program so an
#: offline machine only needs the folder copied over.
TABPFN_WEIGHTS_ROOT = PROJECT_ROOT / "pretrained" / "tabpfn"

#: Start-up and crash logs of the packaged program (it has no console).
LOGS_ROOT = PROJECT_ROOT / "logs"

#: User preferences that are not tied to one model, such as the UI language.
SETTINGS_PATH = PROJECT_ROOT / "settings.json"


def ensure_folders() -> None:
    for folder in (DEFAULT_OUTPUT_ROOT, MODEL_LIBRARY_ROOT, REPORTS_ROOT, TABPFN_WEIGHTS_ROOT):
        folder.mkdir(parents=True, exist_ok=True)


def model_search_roots() -> tuple[Path, ...]:
    """Folders scanned for models: the library plus every Benchmark Run."""

    return (MODEL_LIBRARY_ROOT, DEFAULT_OUTPUT_ROOT)
