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

from i18n import tr

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
            axis.set_xlabel(tr("Actual"))
            axis.set_ylabel(tr("Predicted"))
            axis.legend(fontsize=8, loc="upper left")
        for position in range(len(targets), rows * columns):
            axes[position // columns][position % columns].axis("off")
        figure.suptitle(
            tr("Predicted vs actual — {model}", model=result.model_label), fontsize=12
        )
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
    axes[0].set_title(tr("Residual distribution"))
    axes[0].set_xlabel(tr("Actual − Predicted"))
    axes[0].set_ylabel(tr("Count"))
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    axes[1].set_title(tr("Residual vs predicted"))
    axes[1].set_xlabel(tr("Predicted"))
    axes[1].set_ylabel(tr("Residual"))
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
        ("nmae", tr("Macro NMAE (lower is better)"), [r.metrics.nmae for r in results]),
        (
            "reported_r2",
            tr("Reported R² (higher is better)"),
            [r.metrics.reported_r2 for r in results],
        ),
        ("mape", tr("MAPE % (descriptive only)"), [r.metrics.mape_percent for r in results]),
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
    lines.append("# " + tr("Batch inference and validation report"))
    lines.append("")
    lines.append("- " + tr("Generated: {timestamp}", timestamp=context["generated_at"]))
    lines.append("- " + tr("Source: `{path}`", path=context["source_path"]))
    lines.append(
        "- "
        + tr(
            "Worksheet: `{sheet}` ({rows:,} rows)",
            sheet=context["sheet"],
            rows=context["total_rows"],
        )
    )
    lines.append("- " + tr("Models in this run: {count}", count=len(results)))
    lines.append("")

    lines.append("## " + tr("Evidence boundary"))
    lines.append("")
    if ranked:
        lines.append(
            tr(
                "The metrics below compare each model's predictions against the ground "
                "truth in this table. They are independent generalization evidence only "
                "if these observations **never took part in that model's training or "
                "tuning**. If this table overlaps the training data, read it as a "
                "reproducibility check rather than as new evidence."
            )
        )
    else:
        lines.append(
            tr(
                "No usable ground-truth column was mapped, so this run reports "
                "predictions only and computes no validation metrics."
            )
        )
    lines.append("")
    lines.append(
        tr(
            "Reported R² is clamped at 0: a 0 means the model did not beat the "
            "always-predict-the-mean baseline. Diagnostic R² keeps negative values for "
            "diagnosis. MAPE is descriptive only — it distorts as the true value "
            "approaches 0 — so it is never used for ranking."
        )
    )
    lines.append("")

    if ranked:
        lines.append("## " + tr("Model comparison (ranked by macro NMAE)"))
        lines.append("")
        lines.append(
            "| "
            + " | ".join(
                (
                    tr("Rank"),
                    tr("Model"),
                    tr("Rows"),
                    "NMAE",
                    "MAE",
                    "RMSE",
                    "MAPE %",
                    tr("Reported R²"),
                    tr("Diagnostic R²"),
                )
            )
            + " |"
        )
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
            lines.append("## " + tr("Per-target metrics"))
            lines.append("")
            lines.append(
                "| "
                + " | ".join(
                    (
                        tr("Model"),
                        tr("Target"),
                        "MAE",
                        "RMSE",
                        "NMAE",
                        "MAPE %",
                        tr("Reported R²"),
                        tr("Diagnostic R²"),
                    )
                )
                + " |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _index, row in per_target.iterrows():
                lines.append(
                    f"| {row['model']} | {row['target']} | {row['mae']:.6g} | {row['rmse']:.6g} | "
                    f"{row['nmae']:.6g} | {row['mape_percent']:.2f} | {row['reported_r2']:.4f} | "
                    f"{row['diagnostic_r2']:.4f} |"
                )
            lines.append("")

    lines.append("## " + tr("Data handling audit"))
    lines.append("")
    lines.append(
        "| "
        + " | ".join(
            (
                tr("Model"),
                tr("Table rows"),
                tr("Rows used"),
                tr("Skipped: input missing"),
                tr("Ground truth missing"),
                tr("Status"),
            )
        )
        + " |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        status = result.error or (
            tr("validated") if result.has_validation else tr("predictions only")
        )
        lines.append(
            f"| {result.model_label} | {result.total_rows:,} | {result.used_rows:,} | "
            f"{result.dropped_missing_features:,} | {result.dropped_missing_targets:,} | {status} |"
        )
    lines.append("")

    notes = [(result.model_label, note) for result in results for note in result.notes]
    if notes:
        lines.append("### " + tr("Notes"))
        lines.append("")
        for label, note in notes:
            lines.append(f"- **{label}**：{note}")
        lines.append("")

    if figure_paths:
        lines.append("## " + tr("Figures"))
        lines.append("")
        for path in figure_paths:
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")

    lines.append("## " + tr("Machine-readable appendix"))
    lines.append("")
    lines.append(
        "- `predictions.csv`: "
        + tr("predicted value, actual value and residual per row and target")
    )
    lines.append(
        "- `predictions_wide.csv`: "
        + tr("one row per observation, ready to paste back into the sheet")
    )
    if metric_frame is not None and not metric_frame.empty:
        lines.append("- `metrics.csv`: " + tr("macro and per-target metrics"))
    lines.append(
        "- `manifest.json`: " + tr("runtime, model list and column mapping")
    )
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
