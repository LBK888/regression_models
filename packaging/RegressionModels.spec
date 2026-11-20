# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for RegressionModels.exe (one folder, no console).

Build with packaging\\build_exe.bat, which creates the CUDA build environment
first. Run from the project root:

    .venv-build\\Scripts\\python -m PyInstaller --noconfirm packaging\\RegressionModels.spec
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent

datas = []
binaries = []
hiddenimports = []

# Optional model packages are imported lazily inside factories, and several of
# them ship DLLs or data files (xgboost VERSION, lightgbm lib_lightgbm.dll,
# catboost's extension, tabpfn's configs). Collect them whole.
for package in ("tabpfn", "pytabkit", "tabm", "catboost", "xgboost", "lightgbm", "skrub"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# The run report records these versions through importlib.metadata.
for distribution in (
    "numpy", "pandas", "scikit-learn", "matplotlib", "torch", "PyQt6", "scipy",
    "pydantic", "pydantic-settings", "huggingface-hub", "safetensors", "einops",
):
    datas += copy_metadata(distribution)

# The interface language is imported by name at run time (locales.<code>).
hiddenimports += collect_submodules("locales")
hiddenimports += collect_submodules("regression_v4")
hiddenimports += collect_submodules("app")
hiddenimports += ["legacy_architectures", "regression_core", "startup_checks", "version"]

# Example models: copied next to the exe on first start (startup_checks).
datas += [(str(ROOT / "models" / "legacy_v2"), "seed/models/legacy_v2")]
datas += [(str(ROOT / "LICENSE"), "."), (str(ROOT / "docs" / "third-party-notices.md"), "docs")]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "IPython", "jupyter", "notebook", "pytest",
        "PyQt5", "PySide2", "PySide6",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RegressionModels",
    console=False,
    debug=False,
    strip=False,
    upx=False,  # never UPX-compress CUDA DLLs
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RegressionModels",
)
