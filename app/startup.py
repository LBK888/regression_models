# -*- coding: utf-8 -*-
"""Start-up dialogs: explain a missing GPU and offer the TabPFN weights.

The checks themselves live in ``startup_checks`` (no Qt). This module only
turns their reports into short, actionable messages after the main window is
on screen, so a slow check never looks like a frozen program.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QProgressDialog, QWidget

from i18n import tr
from startup_checks import (
    GpuReport,
    WorkspaceReport,
    download_tabpfn_weights,
    read_settings,
    tabpfn_weights_path,
    update_settings,
)


def gpu_summary(report: GpuReport) -> str:
    """One status-bar phrase for the compute device."""

    if report.usable:
        return tr("GPU ready: {name}", name=report.device_name)
    if report.status == "forced_cpu":
        return tr("CPU mode (GPU disabled by REGRESSION_FORCE_CPU)")
    return tr("CPU mode — no usable GPU")


def gpu_explanation(report: GpuReport) -> str:
    """What happened, what it means, and what the user can do."""

    consequence = tr(
        "The program runs on the CPU instead. Neural networks and TabPFN train more "
        "slowly; every other model is unaffected."
    )
    if report.status == "cpu_build":
        cause = (
            tr(
                "An NVIDIA driver is installed, but this copy of the program contains the "
                "CPU-only PyTorch build ({torch}), so it cannot use the GPU. Use the CUDA "
                "release, or run install.bat for a source installation.",
                torch=report.torch_version,
            )
            if report.driver_found
            else tr("This copy of the program contains the CPU-only PyTorch build.")
        )
    elif report.status == "no_driver":
        cause = tr(
            "No NVIDIA graphics driver was found. If this computer has an NVIDIA card, "
            "install the latest driver from nvidia.com and restart the program."
        )
    elif report.status == "unavailable":
        cause = tr(
            "An NVIDIA driver is installed, but CUDA {cuda} cannot start on it. Update the "
            "NVIDIA driver to the latest version and restart the program.",
            cuda=report.torch_cuda,
        )
    elif report.status == "kernel_failed":
        cause = tr(
            "The GPU {name} was found, but a test calculation failed: {detail}. The card may "
            "be too old or too new for this CUDA {cuda} build, or the driver may be outdated.",
            name=report.device_name or "?",
            detail=report.detail,
            cuda=report.torch_cuda,
        )
    else:
        return ""
    return f"{cause}\n\n{consequence}"


def show_gpu_warning(parent: QWidget, report: GpuReport) -> None:
    if report.usable or report.status == "forced_cpu":
        return
    if read_settings().get("hide_gpu_warning"):
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("No usable GPU"))
    box.setText(tr("CUDA GPU acceleration is not available."))
    box.setInformativeText(gpu_explanation(report))
    remember = QCheckBox(tr("Don't show this again"))
    box.setCheckBox(remember)
    box.exec()
    if remember.isChecked():
        update_settings(hide_gpu_warning=True)


class _DownloadWorker(QThread):
    progressed = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            path = download_tabpfn_weights(self._report)
        except InterruptedError:
            self.failed.emit("")
        except Exception as exc:  # noqa: BLE001 — shown to the user verbatim
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(str(path))

    def _report(self, done: int, total: int) -> bool:
        self.progressed.emit(done, total)
        return not self.isInterruptionRequested()


def offer_tabpfn_download(parent: QWidget, workspace: WorkspaceReport) -> None:
    """Ask once per start whether to fetch the missing TabPFN v2 weights."""

    if workspace.tabpfn_weights == "ok" or read_settings().get("skip_tabpfn_download"):
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("TabPFN weights missing"))
    box.setText(tr("Download the TabPFN v2 weights now (about 45 MB, no account needed)?"))
    box.setInformativeText(
        tr(
            "They are saved to {path}. Without them TabPFN downloads them the first time it "
            "is used, which fails on an offline computer.",
            path=str(tabpfn_weights_path()),
        )
    )
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.button(QMessageBox.StandardButton.Yes).setText(tr("Download"))
    box.button(QMessageBox.StandardButton.No).setText(tr("Not now"))
    remember = QCheckBox(tr("Don't ask again"))
    box.setCheckBox(remember)
    answer = box.exec()
    if remember.isChecked():
        update_settings(skip_tabpfn_download=True)
    if answer != QMessageBox.StandardButton.Yes:
        return

    dialog = QProgressDialog(tr("Downloading the TabPFN v2 weights…"), tr("Cancel"), 0, 100, parent)
    dialog.setWindowTitle(tr("TabPFN weights"))
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    worker = _DownloadWorker(parent)

    def on_progress(done: int, total: int) -> None:
        if total:
            dialog.setValue(min(99, int(done * 100 / total)))
        dialog.setLabelText(
            tr("Downloading the TabPFN v2 weights… {done:.1f} MB", done=done / 1_000_000)
        )

    def on_done(path: str) -> None:
        dialog.setValue(100)
        dialog.close()
        QMessageBox.information(parent, tr("TabPFN weights"), tr("Saved to {path}.", path=path))

    def on_failed(message: str) -> None:
        dialog.close()
        if message:
            QMessageBox.warning(
                parent,
                tr("TabPFN weights"),
                tr(
                    "The download failed: {error}\n\nOn an offline computer, copy "
                    "tabpfn-v2-regressor.ckpt into {folder} by hand.",
                    error=message,
                    folder=str(tabpfn_weights_path().parent),
                ),
            )

    worker.progressed.connect(on_progress)
    worker.finished_ok.connect(on_done)
    worker.failed.connect(on_failed)
    dialog.canceled.connect(worker.requestInterruption)
    worker.start()
    # Keep a reference until the thread ends so it is not garbage collected.
    parent._tabpfn_download_worker = worker  # type: ignore[attr-defined]


def workspace_summary(workspace: WorkspaceReport) -> str:
    parts: list[str] = []
    if workspace.created_folders:
        parts.append(tr("created folders: {names}", names=", ".join(workspace.created_folders)))
    if workspace.settings_created:
        parts.append(tr("created settings.json"))
    if workspace.seeded_models:
        parts.append(tr("installed example models: {names}", names=", ".join(workspace.seeded_models)))
    parts.append(
        tr("TabPFN weights ready") if workspace.tabpfn_weights == "ok" else tr("TabPFN weights missing")
    )
    parts.extend(workspace.problems)
    return "; ".join(parts)
