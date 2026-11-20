# -*- coding: utf-8 -*-
"""Choose the PyTorch wheel index that matches this machine's NVIDIA driver and GPU.

Used by install.bat. Prints one candidate per line, best first, as

    <tag>|<index url>|<label>|<reason>

where <tag> is the CUDA version of the build ("13.0", "12.8", ...) or "cpu".
install.bat tries them in order, because not every build exists for every
Python version (cu118 has no Python 3.14 wheels, for example). The last line is
always the CPU build. Diagnostics go to stderr so they show in the console.

Why not parse nvidia-smi in batch: drivers from the 590 series on print
"CUDA UMD Version: 13.3" where older ones printed "CUDA Version: 12.4", and a
literal findstr for the old text silently selected the CPU build.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

INDEX = "https://download.pytorch.org/whl/{}"

#: (tag, index suffix, minimum driver CUDA, minimum and maximum compute capability)
#: - CUDA 13 builds dropped Maxwell, Pascal and Volta (sm < 7.5).
#: - Recent cu128 builds no longer ship Maxwell/Pascal kernels, and cu128 is the
#:   first build line with Blackwell (RTX 50xx, sm_120) kernels.
#: - cu126 / cu118 have no Blackwell kernels.
BUILDS: tuple[tuple[str, str, tuple[int, int], float, float | None], ...] = (
    ("13.0", "cu130", (13, 0), 7.5, None),
    ("12.8", "cu128", (12, 8), 7.0, None),
    ("12.6", "cu126", (12, 6), 5.0, 9.0),
    ("11.8", "cu118", (11, 8), 3.7, 9.0),
)

#: Fallback when the header shows no CUDA version: minimum Windows driver
#: release for each CUDA line (approximate, by driver branch).
DRIVER_TO_CUDA: tuple[tuple[float, tuple[int, int]], ...] = (
    (580.0, (13, 0)),
    (570.0, (12, 8)),
    (560.0, (12, 6)),
    (522.0, (11, 8)),
)


def parse_cuda_version(header: str) -> tuple[int, int] | None:
    """Highest CUDA version the driver supports, from the nvidia-smi banner.

    Accepts both "CUDA Version: 12.4" (older drivers) and
    "CUDA UMD Version: 13.3" (590 series and newer).
    """

    match = re.search(r"CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)", header)
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_from_driver(driver: str) -> tuple[int, int] | None:
    try:
        version = float(".".join(driver.strip().split(".")[:2]))
    except ValueError:
        return None
    for minimum, cuda in DRIVER_TO_CUDA:
        if version >= minimum:
            return cuda
    return None


def candidates(cuda: tuple[int, int] | None, capability: float | None) -> list[tuple[str, str, str, str]]:
    """Suitable builds, best first, always ending with the CPU build."""

    rows: list[tuple[str, str, str, str]] = []
    if cuda is not None:
        for tag, suffix, needs, low, high in BUILDS:
            if cuda < needs:
                continue
            if capability is not None and (capability < low or (high is not None and capability > high)):
                continue
            reason = f"driver supports CUDA {cuda[0]}.{cuda[1]}"
            if capability is not None:
                reason += f", GPU compute capability {capability:g}"
            rows.append((tag, INDEX.format(suffix), f"CUDA {tag} build", reason))
    rows.append(("cpu", INDEX.format("cpu"), "CPU-only build", "fallback without a usable CUDA build"))
    return rows


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    return str(candidate) if candidate.exists() else None


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def detect() -> tuple[tuple[int, int] | None, float | None, str]:
    smi = _nvidia_smi()
    if smi is None:
        return None, None, "nvidia-smi not found (no NVIDIA driver)"
    header = _run([smi])
    query = _run([smi, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
    gpus = [line.split(",") for line in query.splitlines() if line.strip()]
    names = ", ".join(parts[0].strip() for parts in gpus) or "unknown GPU"
    driver = gpus[0][1].strip() if gpus and len(gpus[0]) > 1 else ""
    capabilities: list[float] = []
    for parts in gpus:
        try:
            capabilities.append(float(parts[2]))
        except (IndexError, ValueError):
            pass
    cuda = parse_cuda_version(header) or (cuda_from_driver(driver) if driver else None)
    # Several GPUs: the build must run on the oldest one.
    capability = min(capabilities) if capabilities else None
    summary = f"{names}; driver {driver or '?'}; CUDA {'.'.join(map(str, cuda)) if cuda else '?'}"
    if capability is not None:
        summary += f"; compute capability {capability:g}"
    return cuda, capability, summary


def main() -> int:
    cuda, capability, summary = detect()
    print(f"      Detected: {summary}", file=sys.stderr)
    for row in candidates(cuda, capability):
        print("|".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
