#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for training, the model library and inference.

Usage
-----
    python main.py                 launch the merged UI
    python main.py --self-test     check the engine without opening a window
    python main.py --validate FILE audit a dataset with the shared loader
    python main.py --scan          list what the model library currently sees
    python main.py --check         run the start-up checks (GPU, folders, weights)
    python main.py --model-check   fit every available model once on synthetic data

The packaged RegressionModels.exe accepts the same options. It has no console,
so their output goes to logs/console.log beside the exe.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import i18n
from i18n import LANGUAGES


def _route_output_to_log() -> None:
    """A windowed exe has no console: print() would fail, so write to a log.

    When a console or pipe *is* attached, switch it to UTF-8: the packaged
    Python ignores PYTHONIOENCODING, and the JSON printed by --scan / --check
    contains Chinese file names that the Windows code page would garble.
    """

    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    if sys.stdout is not None and sys.stderr is not None:
        return
    from project_paths import LOGS_ROOT

    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        stream = open(LOGS_ROOT / "console.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        import io

        stream = io.StringIO()
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _install_crash_log() -> None:
    """Keep a traceback of any unexpected error, and show it when a window exists."""

    import datetime
    import traceback

    from project_paths import LOGS_ROOT

    def hook(kind, value, tb) -> None:  # type: ignore[no-untyped-def]
        text = "".join(traceback.format_exception(kind, value, tb))
        try:
            LOGS_ROOT.mkdir(parents=True, exist_ok=True)
            with open(LOGS_ROOT / "crash.log", "a", encoding="utf-8") as handle:
                handle.write(f"--- {datetime.datetime.now().isoformat(timespec='seconds')}\n{text}\n")
        except OSError:
            pass
        print(text, file=sys.stderr)
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(None, "Regression Models", text[-3000:])
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def _startup_checks():  # type: ignore[no-untyped-def]
    """Folders and settings first (they decide the language), then the GPU."""

    from startup_checks import check_gpu, ensure_workspace, report_as_dict
    from project_paths import LOGS_ROOT

    workspace = ensure_workspace()
    gpu = check_gpu()
    try:
        (LOGS_ROOT / "startup.json").write_text(
            json.dumps(report_as_dict(gpu, workspace), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    return gpu, workspace


def _self_test() -> int:
    """Confirm the model registry and the metric core still line up."""

    import numpy as np

    from regression_v4.metrics import calculate_regression_metrics
    from regression_v4.models import model_registry

    registry = model_registry()
    visible = {model.model_id for model in registry}
    required = {
        "mean",
        "ridge",
        "pls",
        "svr_rbf",
        "random_forest",
        "extra_trees",
        "deep_mlp",
        "wide_mlp",
        "residual_mlp",
        "multi_branch_mlp",
        "bottleneck_mlp",
        "cnn1d",
        "resnet1d",
        "grouped_fusion",
        "mlp_embeddings",
        "ft_transformer",
        "modern_nca",
    }
    missing = required - visible
    if missing:
        print(json.dumps({"status": "error", "missing_models": sorted(missing)}, indent=2))
        return 1

    truth = np.array([[1.0], [2.0], [3.0], [4.0]])
    metrics = calculate_regression_metrics(truth, truth + 0.1, ("check",))
    assert metrics.reported_r2 >= 0.0

    from app.adapters import load_predictor
    from app.model_library import discover_model_files

    assert callable(load_predictor)

    # Report rendering: the PDF backend and the CJK font fallback are the parts
    # most likely to break in a packaged build, so draw one small page.
    import tempfile

    import regression_v4.reporting  # noqa: F401 — applies the report font list
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with tempfile.TemporaryDirectory() as folder:
        pdf_path = Path(folder) / "self_test.pdf"
        with PdfPages(pdf_path) as pdf:
            figure = plt.figure(figsize=(4, 3))
            figure.text(0.1, 0.5, "水質檢測 NH3 R² = 0.10")
            pdf.savefig(figure)
            plt.close(figure)
        report_bytes = pdf_path.stat().st_size
    assert report_bytes > 1000, "report PDF was not rendered"

    print(
        json.dumps(
            {
                "status": "ok",
                "report_rendering": f"ok ({report_bytes} bytes)",
                "active_models": sorted(model.model_id for model in registry if model.enabled),
                "optional_unavailable": sorted(
                    model.model_id for model in registry if not model.enabled
                ),
                "model_files_found": len(discover_model_files()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _model_check() -> int:
    """Fit every available model family on a tiny synthetic table.

    Confirms that each optional package actually imports and trains on this
    machine — the packaged exe in particular — and on which device. Exit code 1
    when any available model fails.
    """

    import time

    import numpy as np

    from regression_core import DEVICE
    from regression_v4.models import create_model, model_registry

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] ** 2 + rng.normal(0, 0.1, len(X))
    layouts = {
        "cnn1d": (tuple(f"{400 + 10 * index}nm" for index in range(6)), "ordered_1d"),
        "resnet1d": (tuple(f"{400 + 10 * index}nm" for index in range(6)), "ordered_1d"),
        "grouped_fusion": (("A::x1", "A::x2", "A::x3", "B::x1", "B::x2", "B::x3"), "grouped"),
    }
    results: dict[str, dict[str, object]] = {}
    failures = 0
    for capability in model_registry():
        if not capability.enabled:
            results[capability.model_id] = {"status": "unavailable", "reason": capability.reason}
            continue
        columns, structure = layouts.get(
            capability.model_id, (tuple(f"x{index}" for index in range(6)), "tabular")
        )
        options = {
            "epochs": 5,
            "patience": 3,
            "batch_size": 16,
            "input_columns": columns,
            "feature_structure": structure,
            "structure_confirmed": True,
            "model_size_mode": "auto",
            "capacity_rows": 48,
        }
        started = time.perf_counter()
        try:
            model = create_model(capability.model_id, 0, 1, options)
            model.fit(X[:48], y[:48])
            prediction = np.asarray(model.predict(X[48:]), dtype=float).reshape(-1)
            if prediction.shape != (16,) or not np.isfinite(prediction).all():
                raise ValueError(f"unexpected prediction {prediction.shape}")
            results[capability.model_id] = {
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 2),
            }
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            failures += 1
            results[capability.model_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
    print(
        json.dumps(
            {"device": str(DEVICE), "failures": failures, "models": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


def _validate(path: str) -> int:
    from regression_core import FlexibleDatasetLoader

    bundle = FlexibleDatasetLoader().load(path)
    summary = {
        "source_path": bundle.source_path,
        "sheets": {
            name: {
                "rows": int(len(sheet.data)),
                "columns": [str(column) for column in sheet.data.columns],
                "trainable_columns": bundle.trainable_columns(name),
                "header_row": sheet.profile.header_row_index,
                "removed_empty_rows": sheet.profile.removed_empty_rows,
                "removed_empty_columns": sheet.profile.removed_empty_columns,
            }
            for name, sheet in bundle.sheets.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _scan() -> int:
    from app.model_library import LibraryIndex, scan_library

    index = LibraryIndex()
    rows = []
    for entry in scan_library(index):
        rows.append(
            {
                "id": entry.entry_id,
                "run": entry.run_label,
                "name": entry.display_name,
                "kind": entry.kind_label,
                "inputs": len(entry.feature_names),
                "outputs": len(entry.target_names),
                "reported_r2": entry.reported_r2,
                "mape_percent": entry.mape_percent,
                "enabled": entry.enabled,
                "compatible": entry.compatible,
                "status": entry.reason,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regression Models — training, model library and inference"
    )
    parser.add_argument("--self-test", action="store_true", help="check the engine without a GUI")
    parser.add_argument("--validate", metavar="DATASET", help="audit a dataset with the shared loader")
    parser.add_argument("--scan", action="store_true", help="list what the model library sees")
    parser.add_argument(
        "--check", action="store_true", help="run the start-up checks (GPU, folders, weights)"
    )
    parser.add_argument(
        "--model-check",
        action="store_true",
        help="fit every available model on a tiny synthetic table and report pass/fail",
    )
    parser.add_argument(
        "--lang",
        choices=[code for code, _name in LANGUAGES],
        help="override the interface language for this launch",
    )
    args = parser.parse_args(argv)

    if args.check:
        from startup_checks import report_as_dict

        gpu, workspace = _startup_checks()
        print(json.dumps(report_as_dict(gpu, workspace), ensure_ascii=False, indent=2))
        return 0

    i18n.initialize()
    if args.lang:
        i18n.set_language(args.lang, persist=False)

    if args.self_test:
        return _self_test()
    if args.validate:
        return _validate(args.validate)
    if args.scan:
        return _scan()
    if args.model_check:
        _startup_checks()  # pick the same device the window would use
        return _model_check()

    gpu, workspace = _startup_checks()
    # ensure_workspace() may have just written the first settings.json with the
    # system language, so read the language again.
    i18n.initialize()
    if args.lang:
        i18n.set_language(args.lang, persist=False)

    from app.main_window import launch

    return launch(sys.argv, gpu_report=gpu, workspace_report=workspace)


if __name__ == "__main__":
    # Required for any multiprocessing child started from the packaged exe.
    multiprocessing.freeze_support()
    _route_output_to_log()
    _install_crash_log()
    raise SystemExit(main())
