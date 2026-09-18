#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for training, the model library and inference.

Usage
-----
    python main.py                 launch the merged UI
    python main.py --self-test     check the engine without opening a window
    python main.py --validate FILE audit a dataset with the shared loader
    python main.py --scan          list what the model library currently sees
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    if "tabpfn" in visible:
        print(json.dumps({"status": "error", "reason": "TabPFN should have been removed"}, indent=2))
        return 1

    truth = np.array([[1.0], [2.0], [3.0], [4.0]])
    metrics = calculate_regression_metrics(truth, truth + 0.1, ("check",))
    assert metrics.reported_r2 >= 0.0

    from app.adapters import load_predictor
    from app.model_library import discover_model_files

    assert callable(load_predictor)

    print(
        json.dumps(
            {
                "status": "ok",
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
    parser = argparse.ArgumentParser(description="Regression Models — 訓練與推論整合介面")
    parser.add_argument("--self-test", action="store_true", help="驗證核心，不開啟 GUI")
    parser.add_argument("--validate", metavar="DATASET", help="用共用載入器稽核一個資料檔")
    parser.add_argument("--scan", action="store_true", help="列出模型庫目前看到的模型")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.validate:
        return _validate(args.validate)
    if args.scan:
        return _scan()

    from app.main_window import launch

    return launch(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
