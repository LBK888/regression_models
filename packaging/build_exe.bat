@echo off
setlocal
REM ======================================================================
REM  Build dist\RegressionModels\RegressionModels.exe (CUDA 12.8 edition)
REM  and the split 7-Zip archive that goes on the GitHub release.
REM
REM  Uses its own .venv-build so the everyday .venv is never touched.
REM  Needs Python 3.14 and, for the archive, 7-Zip in its default folder.
REM ======================================================================
cd /d "%~dp0\.."

set "BUILD_VENV=.venv-build"
set "PY=%BUILD_VENV%\Scripts\python.exe"
set "TORCH_SPEC=torch==2.11.0+cu128"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"

for /f "delims=" %%v in ('python -c "import version;print(version.APP_VERSION)"') do set "APP_VERSION=%%v"
if "%APP_VERSION%"=="" set "APP_VERSION=dev"
set "ARCHIVE=release\RegressionModels-v%APP_VERSION%-win64-cuda12.8.7z"

echo [1/4] Build environment...
if not exist "%PY%" (
    python -m venv "%BUILD_VENV%" || goto :fail
    "%PY%" -m pip install --upgrade pip || goto :fail
    REM CUDA PyTorch first, so tabpfn's "torch>=2.5" does not pull the CPU build.
    "%PY%" -m pip install "%TORCH_SPEC%" --index-url %TORCH_INDEX% || goto :fail
    "%PY%" -m pip install -r requirements.txt pyinstaller || goto :fail
)

echo [2/4] PyInstaller...
"%PY%" -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\RegressionModels.spec || goto :fail

echo [3/4] Extra files...
copy /y packaging\README_release.txt dist\RegressionModels\README.txt >nul
if exist pretrained\tabpfn\tabpfn-v2-regressor.ckpt (
    mkdir dist\RegressionModels\pretrained\tabpfn 2>nul
    copy /y pretrained\tabpfn\tabpfn-v2-regressor.ckpt dist\RegressionModels\pretrained\tabpfn\ >nul
) else (
    echo       TabPFN weights not found locally; the exe will offer to download them.
)

echo [4/4] Archive (1.9 GB volumes, GitHub allows 2 GiB per file)...
if not exist "%SEVENZIP%" (
    echo       7-Zip not found; skipping the archive. dist\RegressionModels is ready.
    goto :done
)
if not exist release mkdir release
if exist "%ARCHIVE%*" (
    echo       %ARCHIVE% already exists - delete it first or raise APP_VERSION.
    goto :fail
)
pushd dist
"%SEVENZIP%" a -t7z -mx=7 -mmt=on -v1900m "..\%ARCHIVE%" RegressionModels || (popd & goto :fail)
popd
REM A single volume is an ordinary .7z: drop the .001 suffix so users get one plain file.
if exist "%ARCHIVE%.001" if not exist "%ARCHIVE%.002" move /y "%ARCHIVE%.001" "%ARCHIVE%" >nul

:done
echo.
echo Done: dist\RegressionModels\RegressionModels.exe
exit /b 0

:fail
echo.
echo [ERROR] Build failed - see the messages above.
exit /b 1
