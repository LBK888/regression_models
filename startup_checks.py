# -*- coding: utf-8 -*-
"""Start-up checks: is CUDA really usable, and is the program folder complete?

Kept free of Qt so the same checks run from the command line
(``RegressionModels.exe --check`` / ``python main.py --check``) and in the
window. The GPU check must run *before* ``regression_core`` is imported,
because that module picks the compute device once, at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

from i18n import DEFAULT_LANGUAGE
from project_paths import (
    FROZEN,
    LOGS_ROOT,
    MODEL_LIBRARY_ROOT,
    SEED_MODELS_ROOT,
    SETTINGS_PATH,
    TABPFN_WEIGHTS_ROOT,
    ensure_folders,
)

TABPFN_WEIGHTS_FILENAME = "tabpfn-v2-regressor.ckpt"
TABPFN_WEIGHTS_URL = (
    "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/" + TABPFN_WEIGHTS_FILENAME
)
#: The real file is ~44 MB; anything far smaller is a failed or partial download.
TABPFN_WEIGHTS_MIN_BYTES = 30_000_000


# --------------------------------------------------------------------- settings
def read_settings() -> dict[str, Any]:
    try:
        payload = json.loads(Path(SETTINGS_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def update_settings(**values: Any) -> None:
    """Merge values into settings.json; a read-only folder is not an error."""

    settings = read_settings()
    settings.update(values)
    try:
        Path(SETTINGS_PATH).write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


# -------------------------------------------------------------------------- GPU
@dataclass
class GpuReport:
    """Outcome of the CUDA check.

    ``status`` is one of ``ok``, ``forced_cpu``, ``cpu_build``, ``no_driver``,
    ``unavailable`` or ``kernel_failed``.
    """

    status: str
    usable: bool
    device_name: str = ""
    torch_version: str = ""
    torch_cuda: str = ""
    driver_found: bool = False
    detail: str = ""


def _nvidia_driver_found() -> bool:
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (
        shutil.which("nvidia-smi") is not None
        or (system32 / "nvidia-smi.exe").exists()
        or (system32 / "nvcuda.dll").exists()
    )


def check_gpu() -> GpuReport:
    """Verify CUDA end to end: build, driver, device, and a real kernel launch.

    ``torch.cuda.is_available()`` alone is not enough: a card that is newer or
    older than the CUDA build reports available and then fails on the first
    operation with "no kernel image is available". So a small matrix product
    is run and synchronised. When the GPU is not usable, ``REGRESSION_FORCE_CPU``
    is set so the rest of the program never tries the GPU.
    """

    import torch

    report = GpuReport(
        status="ok",
        usable=False,
        torch_version=str(torch.__version__),
        torch_cuda=str(torch.version.cuda or ""),
        driver_found=_nvidia_driver_found(),
    )
    if os.environ.get("REGRESSION_FORCE_CPU", "").strip() == "1":
        report.status = "forced_cpu"
        return report
    if not torch.version.cuda:
        report.status = "cpu_build"
    elif not torch.cuda.is_available():
        report.status = "unavailable" if report.driver_found else "no_driver"
    else:
        try:
            report.device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            report.detail = f"compute capability {major}.{minor}"
            matrix = torch.randn(256, 256, device="cuda")
            value = float((matrix @ matrix).sum().item())
            torch.cuda.synchronize()
            if value != value:  # NaN: the kernel ran but produced garbage
                raise RuntimeError("test matrix product returned NaN")
            report.usable = True
        except Exception as exc:  # noqa: BLE001 — any failure means "use the CPU"
            report.status = "kernel_failed"
            report.detail = f"{type(exc).__name__}: {exc}".strip()[:300]
    if not report.usable:
        os.environ["REGRESSION_FORCE_CPU"] = "1"
    return report


# -------------------------------------------------------------------- workspace
@dataclass
class WorkspaceReport:
    created_folders: list[str] = field(default_factory=list)
    settings_created: bool = False
    default_language: str = ""
    seeded_models: list[str] = field(default_factory=list)
    tabpfn_weights: str = ""  # "ok" | "missing"
    tabpfn_weights_path: str = ""
    problems: list[str] = field(default_factory=list)


def tabpfn_weights_path() -> Path:
    return TABPFN_WEIGHTS_ROOT / TABPFN_WEIGHTS_FILENAME


def tabpfn_weights_present() -> bool:
    path = tabpfn_weights_path()
    return path.exists() and path.stat().st_size >= TABPFN_WEIGHTS_MIN_BYTES


def ensure_workspace() -> WorkspaceReport:
    """Create missing folders and settings, and restore the bundled example models."""

    report = WorkspaceReport()
    for folder in (
        MODEL_LIBRARY_ROOT.parent / "runs",
        MODEL_LIBRARY_ROOT,
        MODEL_LIBRARY_ROOT.parent / "reports",
        TABPFN_WEIGHTS_ROOT,
        LOGS_ROOT,
    ):
        if not folder.exists():
            report.created_folders.append(folder.name)
    try:
        ensure_folders()
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.problems.append(f"Cannot create folders next to the program: {exc}")

    settings = read_settings()
    if not Path(SETTINGS_PATH).exists():
        report.settings_created = True
        report.default_language = DEFAULT_LANGUAGE
        update_settings(language=report.default_language)
        settings = read_settings()

    # The exe carries the example models; copy them out once so the user can
    # rename, disable or delete them like any other library model.
    if FROZEN and SEED_MODELS_ROOT.exists() and not settings.get("seed_models_installed"):
        for source in sorted(SEED_MODELS_ROOT.iterdir()):
            target = MODEL_LIBRARY_ROOT / source.name
            if target.exists():
                continue
            try:
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
                report.seeded_models.append(source.name)
            except OSError as exc:
                report.problems.append(f"Cannot copy example model {source.name}: {exc}")
        update_settings(seed_models_installed=True)

    report.tabpfn_weights = "ok" if tabpfn_weights_present() else "missing"
    report.tabpfn_weights_path = str(tabpfn_weights_path())
    return report


def download_tabpfn_weights(
    progress: Callable[[int, int], bool] | None = None,
    *,
    timeout: float = 60.0,
) -> Path:
    """Download the registration-free TabPFN v2 weights into ``pretrained/tabpfn``.

    ``progress(done_bytes, total_bytes)`` is called while downloading; returning
    False cancels. The file is written to ``.part`` first and renamed only when
    complete, so a cancelled or broken download never looks like valid weights.
    """

    import urllib.request

    target = tabpfn_weights_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(TABPFN_WEIGHTS_URL, headers={"User-Agent": "regression-models"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None and progress(done, total) is False:
                    raise InterruptedError("Download cancelled")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if partial.stat().st_size < TABPFN_WEIGHTS_MIN_BYTES:
        partial.unlink(missing_ok=True)
        raise RuntimeError("The downloaded file is too small to be the TabPFN v2 weights")
    partial.replace(target)
    return target


def report_as_dict(gpu: GpuReport, workspace: WorkspaceReport) -> dict[str, Any]:
    return {"gpu": asdict(gpu), "workspace": asdict(workspace), "frozen": FROZEN}
