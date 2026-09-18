"""Bounded master Markdown/PDF report for V4 benchmark evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .domain import EvaluationConfig
from .evaluation import EvaluationResult, rank_within_target_tasks
from .confirmation import ConfirmationResult
from .models import model_registry


class _ResilientPdfPages:
    """Contain PDF backend failures so already-written evidence artifacts survive."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.backend: PdfPages | None = None
        self.error: str | None = None

    def __enter__(self) -> "_ResilientPdfPages":
        try:
            self.backend = PdfPages(self.path)
        except Exception:
            self.error = traceback.format_exc()
        return self

    def savefig(self, *args: object, **kwargs: object) -> None:
        if self.backend is None or self.error is not None:
            return
        try:
            self.backend.savefig(*args, **kwargs)
        except Exception:
            self.error = traceback.format_exc()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc is not None and self.error is None:
            self.error = "".join(traceback.format_exception(exc_type, exc, tb))
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                if self.error is None:
                    self.error = traceback.format_exc()
        return exc is not None


@dataclass(frozen=True)
class ReportArtifacts:
    markdown_path: Path
    pdf_path: Path | None
    results_csv_path: Path
    fold_metrics_csv_path: Path
    prediction_csv_path: Path
    plot_paths: tuple[Path, ...]
    confirmation_csv_path: Path | None = None
    tuning_trials_csv_path: Path | None = None
    confirmation_fold_metrics_csv_path: Path | None = None
    confirmation_prediction_csv_path: Path | None = None
    external_results_csv_path: Path | None = None
    external_prediction_csv_path: Path | None = None
    run_manifest_path: Path | None = None
    dataset_audit_csv_path: Path | None = None
    experiment_catalog_csv_path: Path | None = None
    failures_csv_path: Path | None = None
    rendering_error_path: Path | None = None


@dataclass(frozen=True)
class NoEvidenceReportArtifacts:
    markdown_path: Path
    pdf_path: Path
    model_paths: tuple[Path, ...]


def build_no_evidence_report(
    output_folder: str | Path,
    config: EvaluationConfig,
    model_paths: Sequence[Path],
) -> NoEvidenceReportArtifacts:
    """Document a 0% holdout/final-only run without inventing evaluation scores."""

    output = Path(output_folder).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    markdown = output / "report.md"
    lines = [
        "# V4 Final-Fit Report — No Generalization Evidence",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Evaluation strategy: `{config.strategy.value}`; test ratio: {config.test_size:.1%}.",
        "",
        "All available rows were used for fitting. No holdout or cross-validation predictions were produced, so this report intentionally contains no test R², MAE, RMSE, MAPE, model ranking, or winner claim.",
        "",
        "Training-set fit scores are omitted because they are not generalization evidence.",
        "",
        "## Saved model bundles",
        "",
        *[f"- `{path.name}`" for path in model_paths],
        "",
    ]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    pdf_path = output / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.92, "V4 Final Fit", fontsize=22, weight="bold")
        figure.text(0.08, 0.86, "No generalization evidence", fontsize=16, color="#a23b28")
        figure.text(0.08, 0.79, f"Strategy: {config.strategy.value}; test ratio: {config.test_size:.1%}", fontsize=11)
        figure.text(
            0.08,
            0.70,
            "All available rows were used for fitting. No holdout or cross-validation\n"
            "predictions were generated. Training scores are intentionally omitted.",
            fontsize=11,
            linespacing=1.6,
        )
        figure.text(0.08, 0.58, f"Saved model bundles: {len(model_paths)}", fontsize=11)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return NoEvidenceReportArtifacts(markdown, pdf_path, tuple(model_paths))


class MasterReportBuilder:
    def __init__(
        self,
        results: Sequence[EvaluationResult],
        config: EvaluationConfig,
        confirmation_results: Sequence[ConfirmationResult] = (),
        external_results: Sequence[EvaluationResult] = (),
        dataset_audit_rows: Sequence[Mapping[str, object]] = (),
        experiment_audit_rows: Sequence[Mapping[str, object]] = (),
        failure_rows: Sequence[Mapping[str, object]] = (),
    ) -> None:
        if not results:
            raise ValueError("At least one evaluation result is required")
        self.results = tuple(results)
        self.config = config
        self.confirmation_results = tuple(confirmation_results)
        self.external_results = tuple(external_results)
        self.dataset_audit_rows = tuple(dict(row) for row in dataset_audit_rows)
        self.experiment_audit_rows = tuple(dict(row) for row in experiment_audit_rows)
        self.failure_rows = tuple(dict(row) for row in failure_rows)

    @staticmethod
    def _json_ready(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): MasterReportBuilder._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MasterReportBuilder._json_ready(item) for item in value]
        if hasattr(value, "value"):
            return MasterReportBuilder._json_ready(getattr(value, "value"))
        if isinstance(value, Path):
            return str(value)
        return value

    def _result_rows(
        self,
        results: Sequence[EvaluationResult] | None = None,
    ) -> list[dict[str, object]]:
        selected = self.results if results is None else results
        return [
            {
                "target_task": result.target_task_key,
                "experiment_key": result.experiment.stable_key,
                "experiment": result.experiment.display_name,
                "model_id": result.model_id,
                "model": result.model_name,
                "feature_structure": result.experiment.feature_structure.value,
                "structure_confirmed": result.experiment.structure_confirmed,
                "feature_groups": json.dumps(result.experiment.feature_groups, ensure_ascii=False),
                "evidence": result.evidence_label,
                "configuration": result.configuration,
                "loss": result.loss_id,
                "scaler": result.scaler_id,
                "augmentation": result.augmentation_id,
                "mae": result.metrics.mae,
                "rmse": result.metrics.rmse,
                "nmae": result.metrics.nmae,
                "mape_percent": result.metrics.mape_percent,
                "reported_r2": result.metrics.reported_r2,
                "diagnostic_r2": result.metrics.diagnostic_r2,
            }
            for result in selected
        ]

    def _prediction_rows(
        self,
        results: Sequence[EvaluationResult] | None = None,
    ) -> list[dict[str, object]]:
        selected = self.results if results is None else results
        rows: list[dict[str, object]] = []
        for result in selected:
            for fold in result.folds:
                for row_index, observation_id in enumerate(fold.observation_ids):
                    for target_index, target in enumerate(result.experiment.output_columns):
                        rows.append(
                            {
                                "target_task": result.target_task_key,
                                "experiment_key": result.experiment.stable_key,
                                "model_id": result.model_id,
                                "fold": fold.fold,
                                "repetition": fold.repetition,
                                "observation_id": observation_id,
                                "target": target,
                                "actual": float(fold.y_true[row_index, target_index]),
                                "predicted": float(fold.y_pred[row_index, target_index]),
                                "residual": float(fold.y_true[row_index, target_index] - fold.y_pred[row_index, target_index]),
                            }
                        )
        return rows

    def _fold_metric_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in self.results:
            for fold in result.folds:
                rows.append(
                    {
                        "target_task": result.target_task_key,
                        "experiment_key": result.experiment.stable_key,
                        "model_id": result.model_id,
                        "configuration": result.configuration,
                        "fold": fold.fold,
                        "repetition": fold.repetition,
                        "loss": result.loss_id,
                        "scaler": result.scaler_id,
                        "augmentation": result.augmentation_id,
                        "mae": fold.metrics.mae,
                        "rmse": fold.metrics.rmse,
                        "nmae": fold.metrics.nmae,
                        "mape_percent": fold.metrics.mape_percent,
                        "reported_r2": fold.metrics.reported_r2,
                        "diagnostic_r2": fold.metrics.diagnostic_r2,
                    }
                )
        return rows

    def _confirmation_rows(self) -> list[dict[str, object]]:
        return [
            {
                "target_task": result.target_task_key,
                "candidate_key": result.finalist.candidate_key,
                "experiment_key": result.finalist.experiment.stable_key,
                "model_id": result.finalist.model_id,
                "model": result.finalist.screening_result.model_name,
                "feature_structure": result.finalist.experiment.feature_structure.value,
                "structure_confirmed": result.finalist.experiment.structure_confirmed,
                "feature_groups": json.dumps(result.finalist.experiment.feature_groups, ensure_ascii=False),
                "screening_rank": result.finalist.screening_rank,
                "model_family": result.finalist.model_family,
                "augmentation": result.augmentation_id,
                "outer_evaluations": len(result.outer_folds),
                "observations": result.n_observations,
                "groups": result.n_groups,
                "nmae": result.nmae,
                "nmae_ci_low": result.nmae_ci_low,
                "nmae_ci_high": result.nmae_ci_high,
                "mae": result.metrics.mae,
                "rmse": result.metrics.rmse,
                "mape_percent": result.metrics.mape_percent,
                "reported_r2": result.metrics.reported_r2,
                "diagnostic_r2": result.metrics.diagnostic_r2,
                "per_target_metrics": json.dumps(
                    [dict(row) for row in result.metrics.per_target],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for result in self.confirmation_results
        ]

    def _tuning_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in self.confirmation_results:
            for trial in result.tuning_trials:
                rows.append(
                    {
                        "target_task": result.target_task_key,
                        "candidate_key": result.finalist.candidate_key,
                        "model_id": result.finalist.model_id,
                        "augmentation": result.augmentation_id,
                        "outer_repeat": trial.outer_repeat,
                        "outer_fold": trial.outer_fold,
                        "trial": trial.trial,
                        "parameters": json.dumps(dict(trial.parameters), ensure_ascii=False, sort_keys=True),
                        "mean_inner_nmae": trial.mean_inner_nmae,
                        "effective_inner_folds": trial.effective_inner_folds,
                        "failed": trial.failed,
                    }
                )
        return rows

    def _confirmation_fold_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in self.confirmation_results:
            for fold in result.outer_folds:
                rows.append(
                    {
                        "target_task": result.target_task_key,
                        "candidate_key": result.finalist.candidate_key,
                        "model_id": result.finalist.model_id,
                        "augmentation": result.augmentation_id,
                        "outer_repeat": fold.outer_repeat,
                        "outer_fold": fold.outer_fold,
                        "test_observations": len(fold.observation_ids),
                        "selected_parameters": json.dumps(
                            dict(fold.selected_parameters), ensure_ascii=False, sort_keys=True
                        ),
                        "selected_inner_nmae": fold.selected_inner_nmae,
                        "nmae": fold.nmae,
                        "mae": fold.metrics.mae,
                        "rmse": fold.metrics.rmse,
                        "mape_percent": fold.metrics.mape_percent,
                        "reported_r2": fold.metrics.reported_r2,
                        "diagnostic_r2": fold.metrics.diagnostic_r2,
                    }
                )
        return rows

    def _confirmation_prediction_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in self.confirmation_results:
            for fold in result.outer_folds:
                for row_index, observation_id in enumerate(fold.observation_ids):
                    for target_index, target in enumerate(result.finalist.experiment.output_columns):
                        actual = float(fold.y_true[row_index, target_index])
                        predicted = float(fold.y_pred[row_index, target_index])
                        rows.append(
                            {
                                "target_task": result.target_task_key,
                                "candidate_key": result.finalist.candidate_key,
                                "model_id": result.finalist.model_id,
                                "augmentation": result.augmentation_id,
                                "outer_repeat": fold.outer_repeat,
                                "outer_fold": fold.outer_fold,
                                "observation_id": observation_id,
                                "target": target,
                                "actual": actual,
                                "predicted": predicted,
                                "residual": actual - predicted,
                            }
                        )
        return rows

    def _overview_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        grouped = rank_within_target_tasks(self.results)
        max_rows = max(min(20, len(rows)) for rows in grouped.values())
        figure, axes = plt.subplots(
            len(grouped), 1, figsize=(13, max(5.5, (0.42 * max_rows + 2.2) * len(grouped))), squeeze=False
        )
        for axis, (task_key, ranked) in zip(axes[:, 0], grouped.items()):
            ranked = ranked[:20]
            frame = pd.DataFrame(
                {
                    "label": [f"{index + 1}. {row.model_name} | {row.experiment.display_name[:54]}" for index, row in enumerate(ranked)],
                    "nmae": [row.metrics.nmae for row in ranked],
                    "fold_sd": [float(np.std([fold.metrics.nmae for fold in row.folds], ddof=1)) if len(row.folds) > 1 else 0.0 for row in ranked],
                }
            )
            positions = np.arange(len(frame))
            axis.barh(positions, frame["nmae"], xerr=frame["fold_sd"], capsize=3, color="#3366a3")
            axis.set_yticks(positions, frame["label"], fontsize=8)
            axis.invert_yaxis()
            axis.set_xlabel(f"{ranked[0].evidence_label} macro NMAE (lower is better)")
            suffix = " — showing Top 20" if len(grouped[task_key]) > 20 else ""
            axis.set_title(f"{task_key} — {', '.join(ranked[0].experiment.output_columns)}{suffix}")
            axis.grid(axis="x", alpha=0.25)
        figure.suptitle("Screening overview — ranked separately inside each Target Task")
        figure.tight_layout()
        path = output_dir / "screening_overview.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _overview_plots(self, output_dir: Path) -> list[tuple[Path, plt.Figure]]:
        """Create one readable page per Target Task, each bounded to its Top 20."""

        plots: list[tuple[Path, plt.Figure]] = []
        for task_key, all_rows in rank_within_target_tasks(self.results).items():
            rows = all_rows[:20]
            figure, axis = plt.subplots(figsize=(13, max(6, 0.46 * len(rows) + 2.2)))
            positions = np.arange(len(rows))
            labels = [
                f"{index + 1}. {row.model_name} | {row.experiment.display_name[:58]}"
                for index, row in enumerate(rows)
            ]
            errors = [
                float(np.std([fold.metrics.nmae for fold in row.folds], ddof=1))
                if len(row.folds) > 1
                else 0.0
                for row in rows
            ]
            axis.barh(positions, [row.metrics.nmae for row in rows], xerr=errors, capsize=3, color="#3366a3")
            axis.set_yticks(positions, labels, fontsize=8)
            axis.invert_yaxis()
            axis.set_xlabel(f"{rows[0].evidence_label} macro NMAE (lower is better)")
            suffix = f" — Top 20 of {len(all_rows)}" if len(all_rows) > 20 else f" — {len(all_rows)} configurations"
            axis.set_title(f"Screening overview: {task_key}{suffix}\n{', '.join(rows[0].experiment.output_columns)}")
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            path = output_dir / f"screening_overview_{task_key}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plots.append((path, figure))
        return plots

    def _finalist_plots(self, output_dir: Path) -> list[tuple[Path, plt.Figure]]:
        plots: list[tuple[Path, plt.Figure]] = []
        for task_key, ranked in rank_within_target_tasks(self.results).items():
            finalist = ranked[0]
            truth = np.concatenate([fold.y_true for fold in finalist.folds], axis=0)
            prediction = np.concatenate([fold.y_pred for fold in finalist.folds], axis=0)
            figure, axes = plt.subplots(len(finalist.experiment.output_columns), 2, figsize=(11, 4.2 * len(finalist.experiment.output_columns)), squeeze=False)
            for index, target in enumerate(finalist.experiment.output_columns):
                actual = truth[:, index]
                predicted = prediction[:, index]
                lower = min(float(actual.min()), float(predicted.min()))
                upper = max(float(actual.max()), float(predicted.max()))
                axes[index, 0].scatter(actual, predicted, alpha=0.75, color="#3366a3")
                axes[index, 0].plot([lower, upper], [lower, upper], "--", color="#d95f02")
                axes[index, 0].set(xlabel="Actual", ylabel=f"{finalist.evidence_label} prediction", title=f"{target}: actual vs predicted")
                residual = actual - predicted
                axes[index, 1].scatter(predicted, residual, alpha=0.75, color="#5a8f29")
                axes[index, 1].axhline(0, linestyle="--", color="#d95f02")
                axes[index, 1].set(xlabel="Prediction", ylabel="Residual (actual - predicted)", title=f"{target}: residual pattern")
                for axis in axes[index]:
                    axis.grid(alpha=0.2)
            figure.suptitle(f"Target Task {task_key} finalist: {finalist.model_name}\n{finalist.experiment.display_name}", fontsize=12)
            figure.tight_layout()
            path = output_dir / f"finalist_{task_key}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plots.append((path, figure))
        return plots

    def _metric_comparison_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        grouped = rank_within_target_tasks(self.results)
        figure, axes = plt.subplots(len(grouped), 1, figsize=(13, max(5.5, 5 * len(grouped))), squeeze=False)
        for axis, (task_key, rows) in zip(axes[:, 0], grouped.items()):
            rows = rows[:20]
            positions = np.arange(len(rows))
            width = 0.38
            axis.bar(positions - width / 2, [row.metrics.mae for row in rows], width, label="MAE")
            axis.bar(positions + width / 2, [row.metrics.rmse for row in rows], width, label="RMSE")
            axis.set_xticks(positions, [str(index + 1) for index in range(len(rows))])
            axis.set(xlabel="Rank in Screening overview", ylabel="Error", title=f"{task_key}: MAE and RMSE — Top {len(rows)}")
            axis.legend(); axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "metric_comparison_top20.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _r2_comparison_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        grouped = rank_within_target_tasks(self.results)
        figure, axes = plt.subplots(len(grouped), 1, figsize=(13, max(5.5, 5 * len(grouped))), squeeze=False)
        for axis, (task_key, rows) in zip(axes[:, 0], grouped.items()):
            rows = rows[:20]
            positions = np.arange(len(rows))
            axis.plot(positions, [row.metrics.reported_r2 for row in rows], "o-", label="Reported R²")
            axis.plot(positions, [row.metrics.diagnostic_r2 for row in rows], "s--", label="Diagnostic R²")
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.set_xticks(positions, [str(index + 1) for index in range(len(rows))])
            axis.set(xlabel="Rank in Screening overview", ylabel="R²", title=f"{task_key}: Reported vs Diagnostic R²")
            axis.legend(); axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / "r2_comparison_top20.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _mape_comparison_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        """Show descriptive percentage error without using it for ranking."""

        grouped = rank_within_target_tasks(self.results)
        figure, axes = plt.subplots(len(grouped), 1, figsize=(13, max(5.5, 5 * len(grouped))), squeeze=False)
        for axis, (task_key, rows) in zip(axes[:, 0], grouped.items()):
            rows = rows[:20]
            positions = np.arange(len(rows))
            axis.bar(positions, [row.metrics.mape_percent for row in rows], color="#6a4c93")
            axis.set_xticks(positions, [str(index + 1) for index in range(len(rows))])
            axis.set(
                xlabel="Rank in Screening overview",
                ylabel="MAPE (%)",
                title=f"{task_key}: descriptive percentage error — Top {len(rows)}",
            )
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "mape_comparison_top20.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _fold_stability_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        grouped = rank_within_target_tasks(self.results)
        figure, axes = plt.subplots(len(grouped), 1, figsize=(13, max(5.5, 5 * len(grouped))), squeeze=False)
        for axis, (task_key, rows) in zip(axes[:, 0], grouped.items()):
            rows = rows[:10]
            values = [[fold.metrics.mae for fold in row.folds] for row in rows]
            axis.boxplot(values, tick_labels=[str(index + 1) for index in range(len(rows))], showmeans=True)
            axis.set(xlabel="Rank in Screening overview", ylabel="Fold/repetition MAE", title=f"{task_key}: stability of Top {len(rows)}")
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "fold_stability_top10.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _residual_distribution_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        labels: list[str] = []
        values: list[np.ndarray] = []
        for task_key, rows in rank_within_target_tasks(self.results).items():
            finalist = rows[0]
            truth = np.concatenate([fold.y_true for fold in finalist.folds])
            prediction = np.concatenate([fold.y_pred for fold in finalist.folds])
            for index, target in enumerate(finalist.experiment.output_columns):
                labels.append(f"{task_key[-6:]}\n{target[-24:]}")
                values.append(truth[:, index] - prediction[:, index])
        figure, axis = plt.subplots(figsize=(max(8, 1.25 * len(values)), 6))
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.axhline(0, color="#d95f02", linestyle="--")
        axis.set(title="Finalist residual distributions by Target Task and output", ylabel="Actual - predicted")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "finalist_residual_distributions.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _factor_effect_plot(self, output_dir: Path) -> tuple[Path, plt.Figure]:
        rows = self._result_rows()
        frame = pd.DataFrame(rows)
        best_by_task = frame.groupby("target_task")["nmae"].transform("min").clip(lower=1e-12)
        frame["relative_nmae"] = frame["nmae"] / best_by_task
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        for axis, (column, title) in zip(
            axes.ravel(),
            (
                ("model_id", "Model architecture"),
                ("augmentation", "Augmentation"),
                ("loss", "Loss"),
                ("scaler", "Scaler"),
            ),
        ):
            summary = frame.groupby(column)["relative_nmae"].median().sort_values()
            axis.bar(np.arange(len(summary)), summary.values, color="#5a8f29")
            axis.set_xticks(np.arange(len(summary)), summary.index, rotation=30, ha="right", fontsize=8)
            axis.set(title=f"{title} effect", ylabel="Median NMAE / best NMAE in Target Task")
            axis.axhline(1.0, color="#d95f02", linestyle="--"); axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Configuration-factor comparison (normalized within Target Task)")
        figure.tight_layout()
        path = output_dir / "configuration_factor_effects.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _confirmation_forest_plots(self, output_dir: Path) -> list[tuple[Path, plt.Figure]]:
        """Plot repeated-outer-CV NMAE and its uncertainty per Target Task."""

        plots: list[tuple[Path, plt.Figure]] = []
        task_keys = sorted({row.target_task_key for row in self.confirmation_results})
        for task_key in task_keys:
            rows = sorted(
                (row for row in self.confirmation_results if row.target_task_key == task_key),
                key=lambda row: row.nmae,
            )[:20]
            positions = np.arange(len(rows))
            lower = [max(0.0, row.nmae - row.nmae_ci_low) for row in rows]
            upper = [max(0.0, row.nmae_ci_high - row.nmae) for row in rows]
            labels = [
                f"{index + 1}. {row.finalist.screening_result.model_name} | "
                f"{row.augmentation_id} | {row.finalist.experiment.display_name[:48]}"
                for index, row in enumerate(rows)
            ]
            figure, axis = plt.subplots(figsize=(13, max(6, 0.48 * len(rows) + 2.3)))
            axis.errorbar(
                [row.nmae for row in rows],
                positions,
                xerr=np.asarray([lower, upper]),
                fmt="o",
                color="#3366a3",
                ecolor="#7aa6c9",
                capsize=4,
            )
            axis.set_yticks(positions, labels, fontsize=8)
            axis.invert_yaxis()
            axis.set_xlabel("Repeated outer-CV NMAE (95% CI; lower is better)")
            axis.set_title(f"Confirmation evidence: {task_key} — Top {len(rows)}")
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            path = output_dir / f"confirmation_forest_{task_key}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plots.append((path, figure))
        return plots

    def _augmentation_ablation_plot(self, output_dir: Path) -> tuple[Path, plt.Figure] | None:
        """Show paired outer-fold NMAE deltas relative to each candidate's original data."""

        originals = {
            row.finalist.candidate_key: row
            for row in self.confirmation_results
            if row.augmentation_id == "original"
        }
        labels: list[str] = []
        values: list[list[float]] = []
        for row in self.confirmation_results:
            if row.augmentation_id == "original" or row.finalist.candidate_key not in originals:
                continue
            baseline = originals[row.finalist.candidate_key]
            baseline_folds = {
                (fold.outer_repeat, fold.outer_fold): fold.nmae for fold in baseline.outer_folds
            }
            deltas = [
                fold.nmae - baseline_folds[(fold.outer_repeat, fold.outer_fold)]
                for fold in row.outer_folds
                if (fold.outer_repeat, fold.outer_fold) in baseline_folds
            ]
            if deltas:
                labels.append(
                    f"{row.finalist.screening_result.model_name}\n{row.augmentation_id}\n"
                    f"{row.target_task_key[-8:]}"
                )
                values.append(deltas)
        if not values:
            return None
        figure, axis = plt.subplots(figsize=(max(9, 1.35 * len(values)), 6))
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.axhline(0, color="#d95f02", linestyle="--")
        axis.set_ylabel("Paired outer-fold NMAE delta (augmentation - original)")
        axis.set_title("Confirmation augmentation ablation — negative values favor augmentation")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "confirmation_augmentation_ablation.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _hyperparameter_selection_plot(self, output_dir: Path) -> tuple[Path, plt.Figure] | None:
        labels: list[str] = []
        for result in self.confirmation_results:
            for fold in result.outer_folds:
                parameter_text = json.dumps(dict(fold.selected_parameters), sort_keys=True)
                labels.append(
                    f"{result.finalist.model_id} | {result.augmentation_id} | {parameter_text}"
                )
        if not labels:
            return None
        counts = pd.Series(labels).value_counts().head(20).sort_values()
        figure, axis = plt.subplots(figsize=(13, max(6, 0.45 * len(counts) + 2.0)))
        axis.barh(np.arange(len(counts)), counts.values, color="#5a8f29")
        axis.set_yticks(np.arange(len(counts)), counts.index, fontsize=8)
        axis.set_xlabel("Number of outer folds selecting this parameter set")
        axis.set_title("Confirmation hyperparameter selection frequency — Top 20")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "confirmation_hyperparameter_frequency.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path, figure

    def _confirmation_diagnostic_plots(self, output_dir: Path) -> list[tuple[Path, plt.Figure]]:
        plots: list[tuple[Path, plt.Figure]] = []
        task_keys = sorted({row.target_task_key for row in self.confirmation_results})
        for task_key in task_keys:
            winner = min(
                (row for row in self.confirmation_results if row.target_task_key == task_key),
                key=lambda row: row.nmae,
            )
            truth = np.concatenate([fold.y_true for fold in winner.outer_folds], axis=0)
            prediction = np.concatenate([fold.y_pred for fold in winner.outer_folds], axis=0)
            targets = winner.finalist.experiment.output_columns
            figure, axes = plt.subplots(len(targets), 2, figsize=(11, 4.2 * len(targets)), squeeze=False)
            for index, target in enumerate(targets):
                actual = truth[:, index]
                predicted = prediction[:, index]
                lower = min(float(actual.min()), float(predicted.min()))
                upper = max(float(actual.max()), float(predicted.max()))
                axes[index, 0].scatter(actual, predicted, alpha=0.65, color="#3366a3")
                axes[index, 0].plot([lower, upper], [lower, upper], "--", color="#d95f02")
                axes[index, 0].set(
                    xlabel="Actual",
                    ylabel="Repeated outer-CV prediction",
                    title=f"{target}: actual vs predicted",
                )
                residual = actual - predicted
                axes[index, 1].scatter(predicted, residual, alpha=0.65, color="#5a8f29")
                axes[index, 1].axhline(0, linestyle="--", color="#d95f02")
                axes[index, 1].set(
                    xlabel="Prediction",
                    ylabel="Residual (actual - predicted)",
                    title=f"{target}: residual pattern",
                )
                for axis in axes[index]:
                    axis.grid(alpha=0.2)
            figure.suptitle(
                f"Confirmed leader for {task_key}: {winner.finalist.screening_result.model_name} | "
                f"{winner.augmentation_id}\n{winner.finalist.experiment.display_name}",
                fontsize=12,
            )
            figure.tight_layout()
            path = output_dir / f"confirmation_diagnostics_{task_key}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plots.append((path, figure))
        return plots

    def _external_diagnostic_plots(self, output_dir: Path) -> list[tuple[Path, plt.Figure]]:
        plots: list[tuple[Path, plt.Figure]] = []
        for result_index, result in enumerate(self.external_results, start=1):
            truth = np.concatenate([fold.y_true for fold in result.folds], axis=0)
            prediction = np.concatenate([fold.y_pred for fold in result.folds], axis=0)
            targets = result.experiment.output_columns
            figure, axes = plt.subplots(len(targets), 2, figsize=(11, 4.2 * len(targets)), squeeze=False)
            for target_index, target in enumerate(targets):
                actual = truth[:, target_index]
                predicted = prediction[:, target_index]
                lower = min(float(actual.min()), float(predicted.min()))
                upper = max(float(actual.max()), float(predicted.max()))
                axes[target_index, 0].scatter(actual, predicted, alpha=0.7, color="#3366a3")
                axes[target_index, 0].plot([lower, upper], [lower, upper], "--", color="#d95f02")
                axes[target_index, 0].set(
                    xlabel="External actual",
                    ylabel="Frozen-model prediction",
                    title=f"{target}: external actual vs predicted",
                )
                residual = actual - predicted
                axes[target_index, 1].scatter(predicted, residual, alpha=0.7, color="#5a8f29")
                axes[target_index, 1].axhline(0, linestyle="--", color="#d95f02")
                axes[target_index, 1].set(
                    xlabel="Prediction",
                    ylabel="Residual (actual - predicted)",
                    title=f"{target}: external residual pattern",
                )
                for axis in axes[target_index]:
                    axis.grid(alpha=0.2)
            figure.suptitle(
                f"Independent external validation: {result.model_name}\n{result.experiment.display_name}",
                fontsize=12,
            )
            figure.tight_layout()
            path = output_dir / f"external_validation_diagnostics_{result_index:02d}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plots.append((path, figure))
        return plots

    @staticmethod
    def _paired_confirmation_status(
        winner: ConfirmationResult,
        baseline: ConfirmationResult | None,
    ) -> tuple[str, str]:
        if baseline is None:
            return (
                "Inconclusive",
                "No confirmed Mean/Dummy comparator was available for this Target Task.",
            )
        if winner is baseline or winner.finalist.model_id == "mean":
            return ("Inconclusive", "The Mean/Dummy baseline remains the confirmed leader.")
        baseline_folds = {
            (fold.outer_repeat, fold.outer_fold): (fold.nmae, fold.observation_ids)
            for fold in baseline.outer_folds
        }
        deltas = np.asarray(
            [
                fold.nmae - baseline_folds[(fold.outer_repeat, fold.outer_fold)][0]
                for fold in winner.outer_folds
                if (fold.outer_repeat, fold.outer_fold) in baseline_folds
                and fold.observation_ids == baseline_folds[(fold.outer_repeat, fold.outer_fold)][1]
            ],
            dtype=float,
        )
        if len(deltas) < 2:
            return ("Inconclusive", "Too few paired outer-fold comparisons were available.")
        mean = float(np.mean(deltas))
        half_width = (
            float(student_t.ppf(0.975, df=len(deltas) - 1))
            * float(np.std(deltas, ddof=1))
            / np.sqrt(len(deltas))
        )
        low, high = mean - half_width, mean + half_width
        detail = (
            f"Paired NMAE delta versus Mean/Dummy: {mean:.4g} "
            f"(95% CI {low:.4g} to {high:.4g}; {len(deltas)} outer-fold evaluations)."
        )
        if high < 0:
            return "Supported", detail
        if mean < 0:
            return "Promising but uncertain", detail
        if low > 0:
            return "Worse than baseline", detail
        return "Inconclusive", detail

    def _screening_markdown_legacy(self, plot_paths: Sequence[Path]) -> str:
        ranked = rank_within_target_tasks(self.results)
        lines = [
            "# V4 Small-Data Regression Benchmark Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Evidence boundary",
            "",
            f"Evaluation strategy: `{self.config.strategy.value}` (current phase: screening). {self.config.evidence_note}",
            "Reported R² is clamped to a minimum of 0 for readability. Signed Diagnostic R² is retained in `results.csv`.",
            "MAPE is reported as an intuitive descriptive percentage error, but it is not used for model selection.",
            "Models are ranked only within an identical Target Task (the same output-column set).",
            "",
            "## Screening overview",
            "",
            f"![Screening overview]({plot_paths[0].name})",
            "",
        ]
        for task_key, rows in ranked.items():
            finalist = rows[0]
            same_experiment_baseline = next(
                (
                    row
                    for row in rows
                    if row.model_id == "mean" and row.experiment.stable_key == finalist.experiment.stable_key
                ),
                None,
            )
            if finalist.model_id == "mean":
                evidence_status = "Inconclusive — the mean baseline remains the screening leader."
            elif same_experiment_baseline is None:
                evidence_status = "Inconclusive — no paired mean baseline exists for this exact experiment."
            else:
                improvement = 1.0 - finalist.metrics.mae / max(same_experiment_baseline.metrics.mae, 1e-12)
                paired = [
                    baseline_fold.metrics.mae - finalist_fold.metrics.mae
                    for baseline_fold, finalist_fold in zip(same_experiment_baseline.folds, finalist.folds)
                ]
                consistent = sum(value > 0 for value in paired)
                if improvement <= 0:
                    evidence_status = "Worse than baseline."
                elif consistent == len(paired) and improvement >= 0.05:
                    evidence_status = f"Promising but uncertain — {improvement:.1%} lower aggregate MAE and better in {consistent}/{len(paired)} paired folds; confirmation is still required."
                else:
                    evidence_status = f"Inconclusive — {improvement:.1%} aggregate MAE improvement but only {consistent}/{len(paired)} paired folds improved."
            lines.extend(
                [
                    f"## Target Task `{task_key}`",
                    "",
                    f"Outputs: {', '.join(finalist.experiment.output_columns)}",
                    "",
                    f"Screening leader: **{finalist.model_name}** on **{finalist.experiment.display_name}**, "
                    f"{finalist.evidence_label} MAE {finalist.metrics.mae:.6g}, RMSE {finalist.metrics.rmse:.6g}, "
                    f"MAPE {finalist.metrics.mape_percent:.2f}%, "
                    f"Reported R² {finalist.metrics.reported_r2:.4f} (Diagnostic R² {finalist.metrics.diagnostic_r2:.4f}).",
                    "",
                    f"Evidence Status: **{evidence_status}**",
                    "",
                    "This is screening evidence, not a claim that the architecture is universally superior. Confirmation-stage repeated nested CV is required for a supported final conclusion.",
                    "",
                ]
            )
        lines.extend(["## Figures", ""])
        for path in plot_paths[1:]:
            lines.extend([f"![{path.stem}]({path.name})", ""])
        lines.extend(
            [
                "## Machine-readable appendix",
                "",
                "- `results.csv`: aggregate error metrics, user-facing Reported R², and signed Diagnostic R².",
                "- `fold_metrics.csv`: paired fold-level metrics for model/configuration comparisons.",
                "- `predictions.csv`: fold-level out-of-fold predictions and residuals per target.",
                "",
            ]
        )
        return "\n".join(lines)

    def _markdown(self, plot_paths: Sequence[Path]) -> str:
        ranked = rank_within_target_tasks(self.results)
        lines = [
            "# V4 Small-Data Regression Benchmark Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Evidence boundary",
            "",
            f"Evaluation strategy: `{self.config.strategy.value}`. {self.config.evidence_note}",
            "Reported R² is clamped to a minimum of 0 for readability. Signed Diagnostic R² is retained in `results.csv`.",
            "MAPE is reported as an intuitive descriptive percentage error, but it is not used for model selection.",
            "Models are ranked only within an identical Target Task (the same output-column set).",
            "Screening is used for promotion only; final Evidence Status is derived only from repeated nested Confirmation when that stage was run.",
            "",
            "## Screening overview",
            "",
            f"![Screening overview]({plot_paths[0].name})",
            "",
        ]
        for task_key, rows in ranked.items():
            leader = rows[0]
            lines.extend(
                [
                    f"### Target Task `{task_key}`",
                    "",
                    f"Outputs: {', '.join(leader.experiment.output_columns)}",
                    "",
                    f"Feature Structure: `{leader.experiment.feature_structure.value}`; "
                    f"confirmed: `{leader.experiment.structure_confirmed}`; "
                    f"groups: {', '.join(leader.experiment.feature_groups)}.",
                    "",
                    f"Screening leader: **{leader.model_name}** on **{leader.experiment.display_name}**, "
                    f"{leader.evidence_label} MAE {leader.metrics.mae:.6g}, RMSE {leader.metrics.rmse:.6g}, "
                    f"NMAE {leader.metrics.nmae:.6g}, MAPE {leader.metrics.mape_percent:.2f}%, "
                    f"Reported R² {leader.metrics.reported_r2:.4f} "
                    f"(Diagnostic R² {leader.metrics.diagnostic_r2:.4f}).",
                    "",
                    "Interpretation: promotion evidence only; no final Evidence Status is assigned from screening.",
                    "",
                ]
            )

        lines.extend(["## Independent external validation", ""])
        if self.external_results:
            for result in self.external_results:
                external_rows = sum(len(fold.observation_ids) for fold in result.folds)
                lines.extend(
                    [
                        f"### Target Task `{result.target_task_key}`",
                        "",
                        f"The development-selected configuration **{result.model_name}** was frozen, fit on all "
                        f"development rows, and evaluated once on **{external_rows}** mapped external row(s).",
                        "",
                        f"External MAE {result.metrics.mae:.6g}, RMSE {result.metrics.rmse:.6g}, "
                        f"NMAE {result.metrics.nmae:.6g}, MAPE {result.metrics.mape_percent:.2f}%, "
                        f"Reported R² {result.metrics.reported_r2:.4f} "
                        f"(Diagnostic R² {result.metrics.diagnostic_r2:.4f}).",
                        "",
                        "This is independent performance evidence for the frozen winner; it is not a new model-ranking round.",
                        "",
                    ]
                )
        else:
            lines.extend(["Not run.", ""])

        lines.extend(["## Confirmation evidence", ""])
        if self.confirmation_results:
            task_keys = sorted({row.target_task_key for row in self.confirmation_results})
            for task_key in task_keys:
                task_rows = [row for row in self.confirmation_results if row.target_task_key == task_key]
                winner = min(task_rows, key=lambda row: row.nmae)
                baselines = [
                    row for row in task_rows
                    if row.finalist.model_id == "mean" and row.augmentation_id == "original"
                ]
                baseline = min(baselines, key=lambda row: row.nmae) if baselines else None
                status, comparison = self._paired_confirmation_status(winner, baseline)
                lines.extend(
                    [
                        f"### Target Task `{task_key}`",
                        "",
                        f"Confirmed leader: **{winner.finalist.screening_result.model_name}** with "
                        f"**{winner.augmentation_id}** on **{winner.finalist.experiment.display_name}**.",
                        "",
                        f"Feature Structure: `{winner.finalist.experiment.feature_structure.value}`; "
                        f"confirmed: `{winner.finalist.experiment.structure_confirmed}`; "
                        f"groups: {', '.join(winner.finalist.experiment.feature_groups)}.",
                        "",
                        f"Repeated outer-CV NMAE: **{winner.nmae:.6g}** "
                        f"(95% CI {winner.nmae_ci_low:.6g} to {winner.nmae_ci_high:.6g}); "
                        f"MAE {winner.metrics.mae:.6g}, RMSE {winner.metrics.rmse:.6g}, "
                        f"MAPE {winner.metrics.mape_percent:.2f}%, "
                        f"Reported R² {winner.metrics.reported_r2:.4f} "
                        f"(Diagnostic R² {winner.metrics.diagnostic_r2:.4f}).",
                        "",
                        f"Protocol: {self.config.confirmation_repeats} repeat(s) × "
                        f"{self.config.confirmation_folds} requested outer folds, "
                        f"{self.config.confirmation_inner_folds} requested inner folds, "
                        f"budget {self.config.confirmation_tuning_budget} trial(s) per outer fold.",
                        "",
                        f"Evidence coverage: {winner.n_observations} observation(s)"
                        + (f", {winner.n_groups} group(s)." if winner.n_groups is not None else "."),
                        "",
                        f"Evidence Status: **{status}** — {comparison}",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "Not run. No final Evidence Status is assigned; screening leaders require repeated nested Confirmation.",
                    "",
                ]
            )
        if self.external_results:
            lines.extend(
                [
                    "- `external_results.csv`: metrics for development-selected winners evaluated once on external data.",
                    "- `external_predictions.csv`: external actual values, frozen-model predictions, and residuals.",
                ]
            )

        lines.extend(["## Figures", ""])
        for path in plot_paths[1:]:
            lines.extend([f"![{path.stem}]({path.name})", ""])
        lines.extend(
            [
                "## Machine-readable appendix",
                "",
                "- `results.csv`: aggregate screening metrics, user-facing Reported R², and signed Diagnostic R².",
                "- `fold_metrics.csv`: paired screening fold metrics for model/configuration comparisons.",
                "- `predictions.csv`: screening out-of-fold predictions and residuals per target.",
                "- `run_manifest.json`: immutable configuration, runtime, model availability, evidence counts, and artifact inventory.",
                "- `dataset_audit.csv`: source-sheet and source-column quality audit.",
                "- `experiment_catalog.csv`: selected experiment identity, feature structure, and compatibility metadata.",
                "- `failures.csv`: skipped or failed model/experiment combinations and reasons.",
            ]
        )
        if self.confirmation_results:
            lines.extend(
                [
                    "- `confirmation_results.csv`: repeated outer-CV estimates and 95% confidence intervals.",
                    "- `tuning_trials.csv`: model-specific inner-CV trials, parameters, failures, and selected scores.",
                    "- `confirmation_fold_metrics.csv`: selected parameters and metrics for every outer fold/repeat.",
                    "- `confirmation_predictions.csv`: repeated outer-CV predictions and residuals per observation/target.",
                ]
            )
        lines.append("")
        return "\n".join(lines)

    def build(self, output_folder: str | Path) -> ReportArtifacts:
        output_dir = Path(output_folder).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        results_csv = output_dir / "results.csv"
        fold_metrics_csv = output_dir / "fold_metrics.csv"
        predictions_csv = output_dir / "predictions.csv"
        pd.DataFrame(self._result_rows()).to_csv(results_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._fold_metric_rows()).to_csv(fold_metrics_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._prediction_rows()).to_csv(predictions_csv, index=False, encoding="utf-8-sig")

        confirmation_csv: Path | None = None
        tuning_trials_csv: Path | None = None
        confirmation_fold_metrics_csv: Path | None = None
        confirmation_prediction_csv: Path | None = None
        external_results_csv: Path | None = None
        external_predictions_csv: Path | None = None
        if self.confirmation_results:
            confirmation_csv = output_dir / "confirmation_results.csv"
            tuning_trials_csv = output_dir / "tuning_trials.csv"
            confirmation_fold_metrics_csv = output_dir / "confirmation_fold_metrics.csv"
            confirmation_prediction_csv = output_dir / "confirmation_predictions.csv"
            pd.DataFrame(self._confirmation_rows()).to_csv(confirmation_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(self._tuning_rows()).to_csv(tuning_trials_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(self._confirmation_fold_rows()).to_csv(
                confirmation_fold_metrics_csv, index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(self._confirmation_prediction_rows()).to_csv(
                confirmation_prediction_csv, index=False, encoding="utf-8-sig"
            )
        if self.external_results:
            external_results_csv = output_dir / "external_results.csv"
            external_predictions_csv = output_dir / "external_predictions.csv"
            pd.DataFrame(self._result_rows(self.external_results)).to_csv(
                external_results_csv, index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(self._prediction_rows(self.external_results)).to_csv(
                external_predictions_csv, index=False, encoding="utf-8-sig"
            )

        dataset_audit_csv = output_dir / "dataset_audit.csv"
        experiment_catalog_csv = output_dir / "experiment_catalog.csv"
        failures_csv = output_dir / "failures.csv"
        pd.DataFrame(self.dataset_audit_rows or ({"status": "not supplied"},)).to_csv(
            dataset_audit_csv, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(self.experiment_audit_rows or ({"status": "not supplied"},)).to_csv(
            experiment_catalog_csv, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            self.failure_rows
            or ({"stage": "run", "experiment_key": "", "model_id": "", "reason": "none"},)
        ).to_csv(failures_csv, index=False, encoding="utf-8-sig")

        overview = self._overview_plots(output_dir)
        finalist = self._finalist_plots(output_dir)
        metric_comparison = self._metric_comparison_plot(output_dir)
        mape_comparison = self._mape_comparison_plot(output_dir)
        r2_comparison = self._r2_comparison_plot(output_dir)
        fold_stability = self._fold_stability_plot(output_dir)
        residual_distribution = self._residual_distribution_plot(output_dir)
        factor_effect = self._factor_effect_plot(output_dir)
        confirmation_forest = self._confirmation_forest_plots(output_dir)
        augmentation_ablation = self._augmentation_ablation_plot(output_dir)
        hyperparameter_selection = self._hyperparameter_selection_plot(output_dir)
        confirmation_diagnostics = self._confirmation_diagnostic_plots(output_dir)
        external_diagnostics = self._external_diagnostic_plots(output_dir)
        all_plots = [
            *overview,
            metric_comparison,
            mape_comparison,
            r2_comparison,
            fold_stability,
            residual_distribution,
            factor_effect,
            *finalist,
            *confirmation_forest,
            *([augmentation_ablation] if augmentation_ablation is not None else []),
            *([hyperparameter_selection] if hyperparameter_selection is not None else []),
            *confirmation_diagnostics,
            *external_diagnostics,
        ]
        markdown = output_dir / "report.md"
        markdown.write_text(self._markdown([path for path, _figure in all_plots]), encoding="utf-8")

        pdf_path = output_dir / "report.pdf"
        pdf_writer = _ResilientPdfPages(pdf_path)
        with pdf_writer as pdf:
            title = plt.figure(figsize=(8.27, 11.69))
            title.text(0.08, 0.92, "V4 Small-Data Regression Benchmark", fontsize=20, weight="bold")
            title.text(0.08, 0.86, f"Strategy: {self.config.strategy.value}", fontsize=12)
            title.text(0.08, 0.81, self.config.evidence_note, fontsize=10, wrap=True)
            title.text(0.08, 0.75, "Reported R² >= 0; signed Diagnostic R² remains in the appendix.", fontsize=10)
            title.text(0.08, 0.69, "Ranking is restricted to identical Target Tasks.", fontsize=10)
            title.text(0.08, 0.65, "MAPE is descriptive only and is not used for ranking.", fontsize=10)
            title.text(
                0.08,
                0.59,
                "Confirmation completed." if self.confirmation_results else "Confirmation not run; screening is promotion evidence only.",
                fontsize=10,
                color="#2d6a4f" if self.confirmation_results else "#a23b28",
            )
            pdf.savefig(title)
            plt.close(title)
            if self.confirmation_results:
                task_keys = sorted({row.target_task_key for row in self.confirmation_results})
                for task_key in task_keys:
                    task_rows = [row for row in self.confirmation_results if row.target_task_key == task_key]
                    winner = min(task_rows, key=lambda row: row.nmae)
                    baselines = [
                        row for row in task_rows
                        if row.finalist.model_id == "mean" and row.augmentation_id == "original"
                    ]
                    baseline = min(baselines, key=lambda row: row.nmae) if baselines else None
                    status, comparison = self._paired_confirmation_status(winner, baseline)
                    summary = plt.figure(figsize=(8.27, 11.69))
                    summary.text(0.08, 0.92, f"Confirmation conclusion — {task_key}", fontsize=18, weight="bold")
                    summary.text(0.08, 0.82, f"Evidence Status: {status}", fontsize=14, weight="bold")
                    summary.text(
                        0.08, 0.70,
                        f"Leader: {winner.finalist.screening_result.model_name} | {winner.augmentation_id}",
                        fontsize=10,
                    )
                    summary.text(
                        0.08, 0.65, f"Experiment: {winner.finalist.experiment.display_name}",
                        fontsize=10, wrap=True,
                    )
                    summary.text(
                        0.08, 0.60,
                        f"NMAE: {winner.nmae:.6g} (95% CI {winner.nmae_ci_low:.6g} to {winner.nmae_ci_high:.6g})",
                        fontsize=10,
                    )
                    summary.text(
                        0.08, 0.55, f"MAE: {winner.metrics.mae:.6g}; RMSE: {winner.metrics.rmse:.6g}",
                        fontsize=10,
                    )
                    summary.text(
                        0.08, 0.50, f"MAPE: {winner.metrics.mape_percent:.2f}% (descriptive only)",
                        fontsize=10,
                    )
                    summary.text(
                        0.08, 0.45, f"Outer evaluations: {len(winner.outer_folds)}",
                        fontsize=10,
                    )
                    coverage = f"Observations: {winner.n_observations}"
                    if winner.n_groups is not None:
                        coverage += f"; groups: {winner.n_groups}"
                    summary.text(0.08, 0.40, coverage, fontsize=10)
                    summary.text(0.08, 0.31, comparison, fontsize=10, wrap=True)
                    summary.text(
                        0.08,
                        0.24,
                        "Conclusion is bounded to this Target Task, candidate pool, dataset identity,\n"
                        "and repeated nested-CV protocol. Final-fit training scores are not evidence.",
                        fontsize=10,
                        linespacing=1.6,
                    )
                    pdf.savefig(summary)
                    plt.close(summary)
            for result in self.external_results:
                summary = plt.figure(figsize=(8.27, 11.69))
                summary.text(0.08, 0.92, "Independent External Validation", fontsize=18, weight="bold")
                summary.text(0.08, 0.84, f"Target Task: {result.target_task_key}", fontsize=10, wrap=True)
                summary.text(0.08, 0.77, f"Frozen winner: {result.model_name}", fontsize=13, weight="bold")
                summary.text(0.08, 0.70, result.experiment.display_name, fontsize=9, wrap=True)
                summary.text(0.08, 0.61, f"MAE: {result.metrics.mae:.6g}", fontsize=11)
                summary.text(0.08, 0.56, f"RMSE: {result.metrics.rmse:.6g}", fontsize=11)
                summary.text(0.08, 0.51, f"NMAE: {result.metrics.nmae:.6g}", fontsize=11)
                summary.text(0.08, 0.46, f"MAPE: {result.metrics.mape_percent:.2f}% (descriptive only)", fontsize=11)
                summary.text(0.08, 0.41, f"Reported R²: {result.metrics.reported_r2:.4f}", fontsize=11)
                summary.text(0.08, 0.36, f"Diagnostic R²: {result.metrics.diagnostic_r2:.4f}", fontsize=11)
                summary.text(
                    0.08,
                    0.24,
                    "External targets were opened only after the development winner was frozen.\n"
                    "These rows were not used for screening, tuning, scaling, augmentation, or fitting.",
                    fontsize=10,
                    linespacing=1.6,
                )
                pdf.savefig(summary)
                plt.close(summary)
            for _path, figure in all_plots:
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        rendering_error_path: Path | None = None
        rendered_pdf_path: Path | None = pdf_path
        if pdf_writer.error is not None:
            rendering_error_path = output_dir / "report_rendering_error.txt"
            rendering_error_path.write_text(
                "PDF rendering failed. CSV, Markdown, PNG, audit, and fitted-model artifacts were preserved.\n\n"
                + pdf_writer.error,
                encoding="utf-8",
            )
            if pdf_path.exists():
                pdf_path.unlink()
            rendered_pdf_path = None

        runtime_packages: dict[str, str] = {}
        for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "torch", "PyQt6", "scipy"):
            try:
                runtime_packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                runtime_packages[package] = "not installed"
        run_manifest = output_dir / "run_manifest.json"
        artifact_inventory = [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != run_manifest
        ]
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "configuration": self._json_ready(asdict(self.config)),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": runtime_packages,
            },
            "model_registry": [
                {
                    "model_id": capability.model_id,
                    "display_name": capability.display_name,
                    "availability": capability.availability.value,
                    "enabled": capability.enabled,
                    "reason": capability.reason,
                }
                for capability in model_registry()
            ],
            "evidence_counts": {
                "screening_results": len(self.results),
                "confirmation_results": len(self.confirmation_results),
                "external_results": len(self.external_results),
                "recorded_failures_or_skips": len(self.failure_rows),
            },
            "pdf_status": "rendered" if rendered_pdf_path is not None else "failed_preserved_other_artifacts",
            "artifacts": artifact_inventory,
        }
        run_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ReportArtifacts(
            markdown,
            rendered_pdf_path,
            results_csv,
            fold_metrics_csv,
            predictions_csv,
            tuple(path for path, _ in all_plots),
            confirmation_csv,
            tuning_trials_csv,
            confirmation_fold_metrics_csv,
            confirmation_prediction_csv,
            external_results_csv,
            external_predictions_csv,
            run_manifest,
            dataset_audit_csv,
            experiment_catalog_csv,
            failures_csv,
            rendering_error_path,
        )
