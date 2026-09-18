# -*- coding: utf-8 -*-
"""Validation report for a batch run, mirroring the Benchmark Run's analysis.

The analysis items follow ``regression_v4.reporting`` so the two reports can be
read side by side:

* an evidence boundary statement, because a batch score is only evidence when
  the rows were never used to fit the model,
* a model comparison table ranked by macro NMAE, with MAE, RMSE, MAPE,
  Reported R² and Diagnostic R²,
* per-target metrics for every model,
* predicted-versus-actual panels with the 1:1 reference line,
* residual distributions and residual-versus-predicted panels, and
* the machine-readable CSV/JSON appendix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .batch import BatchResult, rank_results


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class BatchReportArtifacts:
    folder: Path
    markdown_path: Path
    predictions_path: Path
    wide_predictions_path: Path
    metrics_path: Path | None
    manifest_path: Path
    figure_paths: tuple[Path, ...]
    pdf_path: Path | None
    pdf_error: str | None = None


def _figure(width: float = 9.0, height: float = 5.0) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(width, height))
    return figure, axes


def _scatter_panels(results: Sequence[BatchResult], folder: Path) -> list[tuple[Path, plt.Figure]]:
    """Predicted versus actual, one panel per target, per validated model."""

    produced: list[tuple[Path, plt.Figure]] = []
    for index, result in enumerate(results, start=1):
        targets = result.target_names
        columns = min(3, len(targets))
        rows = int(np.ceil(len(targets) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.3 * rows), squeeze=False)
        for position, target in enumerate(targets):
            axis = axes[position // columns][position % columns]
            actual = result.ground_truth[:, position]
            predicted = result.predictions[:, position]
            axis.scatter(actual, predicted, s=26, alpha=0.75, edgecolor="none")
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            padding = (high - low) * 0.05 or 1.0
            axis.plot([low - padding, high + padding], [low - padding, high + padding],
                      linestyle="--", linewidth=1.0, color="#888888", label="1:1")
            per_target = result.metrics.per_target[position]
            axis.set_title(
                f"{target}\nR²={per_target['reported_r2']:.4f} · MAE={per_target['mae']:.4g}",
                fontsize=10,
            )
            axis.set_xlabel("Actual")
            axis.set_ylabel("Predicted")
            axis.legend(fontsize=8, loc="upper left")
        for position in range(len(targets), rows * columns):
            axes[position // columns][position % columns].axis("off")
        figure.suptitle(f"Predicted vs actual — {result.model_label}", fontsize=12)
        figure.tight_layout()
        path = folder / f"predicted_vs_actual_{index:02d}.png"
        figure.savefig(path, dpi=150)
        produced.append((path, figure))
    return produced


def _residual_panels(results: Sequence[BatchResult], folder: Path) -> list[tuple[Path, plt.Figure]]:
    """Residual distribution and residual-versus-predicted for each model."""

    produced: list[tuple[Path, plt.Figure]] = []
    if not results:
        return produced

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for result in results:
        residuals = (result.ground_truth - result.predictions).ravel()
        axes[0].hist(residuals, bins=min(30, max(6, residuals.size // 3)), alpha=0.5,
                     label=result.model_label)
        axes[1].scatter(result.predictions.ravel(), residuals, s=20, alpha=0.6,
                        label=result.model_label, edgecolor="none")
    axes[0].axvline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Actual − Predicted")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    axes[1].set_title("Residual vs predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    path = folder / "residual_diagnostics.png"
    figure.savefig(path, dpi=150)
    produced.append((path, figure))
    return produced


def _comparison_bars(results: Sequence[BatchResult], folder: Path) -> list[tuple[Path, plt.Figure]]:
    """NMAE, R² and MAPE bars across models, matching the run report's trio."""

    produced: list[tuple[Path, plt.Figure]] = []
    if len(results) < 1:
        return produced
    labels = [result.model_label for result in results]
    positions = np.arange(len(labels))
    for key, title, values in (
        ("nmae", "Macro NMAE (lower is better)", [r.metrics.nmae for r in results]),
        ("reported_r2", "Reported R² (higher is better)", [r.metrics.reported_r2 for r in results]),
        ("mape", "MAPE % (descriptive only)", [r.metrics.mape_percent for r in results]),
    ):
        figure, axis = _figure(max(7.0, 1.6 * len(labels)), 4.6)
        axis.bar(positions, values, color="#5b6ef5")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        axis.set_title(title)
        axis.grid(axis="y", linewidth=0.4, alpha=0.4)
        figure.tight_layout()
        path = folder / f"comparison_{key}.png"
        figure.savefig(path, dpi=150)
        produced.append((path, figure))
    return produced


def _markdown(
    results: Sequence[BatchResult],
    ranked: Sequence[BatchResult],
    metric_frame: pd.DataFrame | None,
    figure_paths: Sequence[Path],
    context: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# 批次推論與驗證報告")
    lines.append("")
    lines.append(f"- 產生時間：{context['generated_at']}")
    lines.append(f"- 資料來源：`{context['source_path']}`")
    lines.append(f"- 工作表：`{context['sheet']}`（共 {context['total_rows']:,} 列）")
    lines.append(f"- 參與模型：{len(results)} 個")
    lines.append("")

    lines.append("## 證據界線")
    lines.append("")
    if ranked:
        lines.append(
            "以下指標是模型對本表格的預測與表格內 ground truth 的比較。"
            "只有當這些觀測值**從未參與該模型的訓練或調參**時，這些數字才構成獨立的泛化證據；"
            "若本表格與訓練資料重疊，請把它視為重現性檢查而非新的證據。"
        )
    else:
        lines.append(
            "本次沒有可用的 ground truth 欄位，因此只輸出預測值，不計算任何驗證指標。"
        )
    lines.append("")
    lines.append(
        "Reported R² 已在 0 截斷：0 代表模型沒有勝過「一律預測平均值」的基準。"
        "Diagnostic R² 保留負值供診斷。MAPE 只作描述用，當真值接近 0 時會失真，因此不作為排名依據。"
    )
    lines.append("")

    if ranked:
        lines.append("## 模型比較（依 macro NMAE 排名）")
        lines.append("")
        lines.append("| 排名 | 模型 | 列數 | NMAE | MAE | RMSE | MAPE % | Reported R² | Diagnostic R² |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rank, result in enumerate(ranked, start=1):
            metrics = result.metrics
            lines.append(
                f"| {rank} | {result.model_label} | {result.used_rows:,} | {metrics.nmae:.6g} | "
                f"{metrics.mae:.6g} | {metrics.rmse:.6g} | {metrics.mape_percent:.2f} | "
                f"{metrics.reported_r2:.4f} | {metrics.diagnostic_r2:.4f} |"
            )
        lines.append("")

    if metric_frame is not None and not metric_frame.empty:
        per_target = metric_frame[metric_frame["target"] != "ALL (macro)"]
        if not per_target.empty:
            lines.append("## 各目標指標")
            lines.append("")
            lines.append("| 模型 | 目標 | MAE | RMSE | NMAE | MAPE % | Reported R² | Diagnostic R² |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _index, row in per_target.iterrows():
                lines.append(
                    f"| {row['model']} | {row['target']} | {row['mae']:.6g} | {row['rmse']:.6g} | "
                    f"{row['nmae']:.6g} | {row['mape_percent']:.2f} | {row['reported_r2']:.4f} | "
                    f"{row['diagnostic_r2']:.4f} |"
                )
            lines.append("")

    lines.append("## 資料處理稽核")
    lines.append("")
    lines.append("| 模型 | 表格列數 | 實際使用 | 輸入缺失略過 | Ground truth 缺失 | 狀態 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        status = result.error or ("已驗證" if result.has_validation else "僅預測")
        lines.append(
            f"| {result.model_label} | {result.total_rows:,} | {result.used_rows:,} | "
            f"{result.dropped_missing_features:,} | {result.dropped_missing_targets:,} | {status} |"
        )
    lines.append("")

    notes = [(result.model_label, note) for result in results for note in result.notes]
    if notes:
        lines.append("### 備註")
        lines.append("")
        for label, note in notes:
            lines.append(f"- **{label}**：{note}")
        lines.append("")

    if figure_paths:
        lines.append("## 圖表")
        lines.append("")
        for path in figure_paths:
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")

    lines.append("## 機器可讀附錄")
    lines.append("")
    lines.append("- `predictions.csv`：每列每個目標的預測值、真值與殘差")
    lines.append("- `predictions_wide.csv`：每個觀測一列，可直接貼回原表")
    if metric_frame is not None and not metric_frame.empty:
        lines.append("- `metrics.csv`：macro 與各目標指標")
    lines.append("- `manifest.json`：執行環境、模型清單與欄位對應")
    lines.append("")
    return "\n".join(lines)


def write_report(
    results: Sequence[BatchResult],
    folder: str | Path,
    *,
    source_path: str,
    sheet: str,
    total_rows: int,
    mappings: dict[str, Any] | None = None,
) -> BatchReportArtifacts:
    """Write every artifact for one batch run into ``folder``."""

    output = Path(folder).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    usable = [result for result in results if not result.error and result.used_rows]
    ranked = rank_results(usable)

    prediction_frames = [result.predictions_frame() for result in usable]
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    predictions_path = output / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    wide_frames = []
    for result in usable:
        frame = result.wide_frame()
        frame.insert(0, "model", result.model_label)
        wide_frames.append(frame)
    wide = pd.concat(wide_frames, ignore_index=True) if wide_frames else pd.DataFrame()
    wide_path = output / "predictions_wide.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    metric_rows = [row for result in usable for row in result.metric_rows()]
    metric_frame = pd.DataFrame(metric_rows) if metric_rows else None
    metrics_path: Path | None = None
    if metric_frame is not None and not metric_frame.empty:
        metrics_path = output / "metrics.csv"
        metric_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    figures: list[tuple[Path, plt.Figure]] = []
    if ranked:
        figures.extend(_comparison_bars(ranked, output))
        figures.extend(_scatter_panels(ranked, output))
        figures.extend(_residual_panels(ranked, output))

    figure_paths = tuple(path for path, _figure_object in figures)
    markdown = _markdown(
        results,
        ranked,
        metric_frame,
        figure_paths,
        {
            "generated_at": generated_at,
            "source_path": source_path,
            "sheet": sheet,
            "total_rows": total_rows,
        },
    )
    markdown_path = output / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    pdf_path: Path | None = output / "report.pdf"
    pdf_error: str | None = None
    if figures:
        try:
            with PdfPages(pdf_path) as pdf:
                for _path, figure in figures:
                    pdf.savefig(figure)
        except Exception as exc:
            pdf_error = f"{type(exc).__name__}: {exc}"
            pdf_path = None
    else:
        pdf_path = None
    for _path, figure in figures:
        plt.close(figure)

    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_path": source_path,
        "sheet": sheet,
        "total_rows": total_rows,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "mappings": mappings or {},
        "models": [
            {
                "model": result.model_label,
                "feature_names": list(result.feature_names),
                "target_names": list(result.target_names),
                "used_rows": result.used_rows,
                "dropped_missing_features": result.dropped_missing_features,
                "dropped_missing_targets": result.dropped_missing_targets,
                "error": result.error,
                "notes": list(result.notes),
                "metrics": result.metrics.to_dict() if result.metrics else None,
            }
            for result in results
        ],
        "pdf_error": pdf_error,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    return BatchReportArtifacts(
        folder=output,
        markdown_path=markdown_path,
        predictions_path=predictions_path,
        wide_predictions_path=wide_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
        pdf_path=pdf_path,
        pdf_error=pdf_error,
    )
