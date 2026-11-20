@echo off
setlocal enabledelayedexpansion
title Regression Models - Environment Installer
cd /d "%~dp0"

REM ===================== configurable =====================
set "VENV_DIR=.venv"
REM Install into another folder instead, e.g. to test the installer without
REM touching .venv:  set REGRESSION_VENV_DIR=.venv-test  before running.
if defined REGRESSION_VENV_DIR set "VENV_DIR=%REGRESSION_VENV_DIR%"
set "REQ_FILE=requirements.txt"
REM Pin a version here for reproducibility, e.g. set "TORCH_SPEC=torch==2.11.0"
REM (the version must exist on the CUDA index that gets selected).
set "TORCH_SPEC=torch"
REM Force a specific interpreter if auto-detection picks a bad one,
REM e.g. set "PY_OVERRIDE=py -3.12"   (leave empty for auto-detect)
set "PY_OVERRIDE="
REM ========================================================

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ==========================================
echo   Regression Models - Environment Installer
echo ==========================================
echo.

REM ------------------------------------------------ [1/6] locate Python
echo [1/6] Looking for Python 3.11 or newer...
set "PY_CMD="
if defined PY_OVERRIDE call :probe_python "%PY_OVERRIDE%"
if defined PY_OVERRIDE if not defined PY_CMD goto :err_override
call :probe_python "python"
call :probe_python "py -3"
call :probe_python "py -3.14"
call :probe_python "py -3.13"
call :probe_python "py -3.12"
call :probe_python "py -3.11"
if not defined PY_CMD goto :err_no_python

for /f "delims=" %%v in ('%PY_CMD% -c "import sys;print(sys.version.split()[0])"') do set "PY_VER=%%v"
echo       OK - Python !PY_VER!   [command: %PY_CMD%]
echo.

REM ------------------------------------------- [2/6] virtual environment
echo [2/6] Preparing virtual environment "%VENV_DIR%"...
if exist "%VENV_PY%" goto :venv_reuse
if not exist "%VENV_DIR%" goto :venv_create

echo       [WARN] "%VENV_DIR%" exists but contains no usable Python interpreter.
set "ANS="
set /p "ANS=      Delete and recreate it? [y/N] "
if /i not "!ANS!"=="y" goto :err_aborted
rd /s /q "%VENV_DIR%"

:venv_create
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :err_venv
if not exist "%VENV_PY%" goto :err_venv
echo       OK - created.
goto :venv_done

:venv_reuse
echo       OK - existing environment reused.

:venv_done
echo.

REM ------------------------------------------------------ [3/6] pip tools
echo [3/6] Upgrading pip / setuptools / wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :err_pip
echo.

REM ------------------------------------------------- [4/6] PyTorch build
REM PyTorch comes first. PyPI only has CPU-only torch for Windows, so if the
REM requirements went first, tabpfn / lightning would pull that CPU build in and
REM the CUDA install afterwards would find torch "already satisfied".
echo [4/6] Choosing the PyTorch build for this machine...
if not exist "tools\select_torch_build.py" goto :err_no_selector
set "TORCH_DONE="
set "TORCH_LABEL="
set "TORCH_TAG="
for /f "usebackq tokens=1-4 delims=|" %%a in (`call "%VENV_PY%" tools\select_torch_build.py`) do (
    if not defined TORCH_DONE call :try_torch "%%a" "%%b" "%%c" "%%d"
)
if not defined TORCH_DONE goto :err_torch
echo       PyTorch: %TORCH_LABEL%
echo.

REM --------------------------------------------------- [5/6] requirements
echo [5/6] Installing project dependencies from %REQ_FILE%...
if not exist "%REQ_FILE%" goto :err_no_req
set "TORCH_PIN="
for /f "usebackq delims=" %%v in (`call "%VENV_PY%" -c "import torch; print(torch.__version__)"`) do set "TORCH_PIN=%%v"
if not defined TORCH_PIN goto :err_verify
REM Pin the exact torch just installed: a package that asked for another torch
REM would otherwise replace the CUDA build with PyPI's CPU-only one.
set "TORCH_CONSTRAINT=%TEMP%\regression_models_torch_pin.txt"
> "%TORCH_CONSTRAINT%" echo torch==%TORCH_PIN%
"%VENV_PY%" -m pip install -r "%REQ_FILE%" -c "%TORCH_CONSTRAINT%"
if errorlevel 1 goto :err_req
echo.

REM ------------------------------------------------- [6/6] verify + folders
echo [6/6] Verifying the installation...
"%VENV_PY%" -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib, openpyxl, PyQt6.QtWidgets, torch; print('  numpy        ', numpy.__version__); print('  pandas       ', pandas.__version__); print('  scikit-learn ', sklearn.__version__); print('  matplotlib   ', matplotlib.__version__); print('  PyQt6        ', 'ok'); print('  torch        ', torch.__version__); print('  CUDA available', torch.cuda.is_available()); print('  GPU          ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
if errorlevel 1 goto :err_verify
if /i "%TORCH_TAG%"=="cpu" goto :verify_selftest
REM is_available() is not enough: run a real calculation on the GPU.
"%VENV_PY%" -c "import torch; x = torch.randn(256, 256, device='cuda'); y = (x @ x).sum().item(); torch.cuda.synchronize(); print('  GPU test      ok -', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :warn_gpu
goto :verify_selftest

:warn_gpu
echo       [WARN] The %TORCH_LABEL% is installed but a test calculation on the GPU failed.
echo              Update the NVIDIA driver to the latest version. Until then the
echo              program runs on the CPU and tells you why at start-up.

:verify_selftest

echo.
echo       Running the application self-test...
"%VENV_PY%" main.py --self-test
if errorlevel 1 goto :err_selftest

echo.
echo ==========================================
echo   Installation complete.
echo   Run start.bat to launch the program.
echo ==========================================
echo.
echo   Extra model adapters (CatBoost, XGBoost, LightGBM, TabM, RealMLP,
echo   TabPFN v2) are installed from requirements.txt. TabPFN v2 downloads
echo   its weights (~45 MB, no account needed) into pretrained\tabpfn the
echo   first time it is used.
echo.
pause
exit /b 0

REM ============================== errors ==============================
:err_no_python
echo.
echo [ERROR] No Python 3.11 or newer was found on this system.
echo         Install it from https://www.python.org/downloads/
echo         and make sure "Add python.exe to PATH" is ticked.
pause
exit /b 1

:err_override
echo.
echo [ERROR] PY_OVERRIDE is set to "%PY_OVERRIDE%" but that interpreter is
echo         missing or older than Python 3.11.
echo         Fix or clear PY_OVERRIDE at the top of install.bat.
pause
exit /b 1

:err_aborted
echo.
echo [ABORTED] Nothing was changed. Remove or rename "%VENV_DIR%" and run install.bat again.
pause
exit /b 1

:err_venv
echo.
echo [ERROR] Failed to create the virtual environment in "%VENV_DIR%".
echo         Check disk space and folder write permissions.
pause
exit /b 1

:err_pip
echo.
echo [ERROR] Failed to upgrade pip / setuptools / wheel. Check your network connection.
pause
exit /b 1

:err_no_req
echo.
echo [ERROR] %REQ_FILE% not found next to install.bat.
pause
exit /b 1

:err_req
echo.
echo [ERROR] Dependency installation failed. Check your network connection or %REQ_FILE%.
echo         If pip reports a conflict with torch==%TORCH_PIN%, a package needs a
echo         different PyTorch version: pin one in TORCH_SPEC at the top of install.bat.
echo         Tip: brand-new Python releases often have no wheels yet - set
echo              PY_OVERRIDE at the top of install.bat to an older 3.1x and retry.
pause
exit /b 1

:err_no_selector
echo.
echo [ERROR] tools\select_torch_build.py is missing next to install.bat.
echo         Copy the whole project folder, including the tools folder.
pause
exit /b 1

:err_torch
echo.
echo [ERROR] PyTorch installation failed - not even the CPU build could be installed.
echo         The build may not exist for this Python version, or the download failed.
echo         Check https://pytorch.org/get-started/locally/ for a matching command.
echo         Tip: brand-new Python releases often have no wheels yet - set
echo              PY_OVERRIDE at the top of install.bat to an older 3.1x and retry.
pause
exit /b 1

:err_verify
echo.
echo [ERROR] The environment was built but at least one package failed to import.
echo         Scroll up for the traceback.
pause
exit /b 1

:err_selftest
echo.
echo [ERROR] Packages installed, but the application self-test failed.
echo         Scroll up for the traceback and report it with the message above.
pause
exit /b 1

REM ============================ subroutines ============================
REM One PyTorch candidate from tools\select_torch_build.py:
REM   %1 = CUDA tag of the build ("13.0", "12.8", ... or "cpu"), %2 = index URL,
REM   %3 = label, %4 = why it was chosen. Sets TORCH_DONE when torch is in place.
:try_torch
set "CAND_TAG=%~1"
set "CAND_INDEX=%~2"
set "CAND_LABEL=%~3"
echo       Candidate: %~3 - %~4
set "HAVE_TAG=none"
for /f "usebackq delims=" %%t in (`call "%VENV_PY%" -c "import torch; print(torch.version.cuda or 'cpu')" 2^>nul`) do set "HAVE_TAG=%%t"
if /i "%HAVE_TAG%"=="%CAND_TAG%" goto :torch_keep
if /i "%CAND_TAG%"=="cpu" if /i not "%HAVE_TAG%"=="none" goto :torch_keep_cuda
if /i "%HAVE_TAG%"=="none" goto :torch_fetch
echo       Replacing the installed PyTorch [%HAVE_TAG%] with the %CAND_LABEL%...
"%VENV_PY%" -m pip uninstall -y torch

:torch_fetch
"%VENV_PY%" -m pip install %TORCH_SPEC% --index-url %CAND_INDEX%
if errorlevel 1 goto :torch_next
goto :torch_ok

:torch_next
echo       [WARN] The %CAND_LABEL% is not available for this Python - trying the next option.
goto :eof

:torch_keep_cuda
echo       Keeping the installed PyTorch [CUDA %HAVE_TAG%]; it also runs on the CPU.
set "CAND_LABEL=CUDA %HAVE_TAG% build"
set "CAND_TAG=%HAVE_TAG%"
goto :torch_ok

:torch_keep
echo       The %CAND_LABEL% is already installed - keeping it.

:torch_ok
set "TORCH_DONE=1"
set "TORCH_LABEL=%CAND_LABEL%"
set "TORCH_TAG=%CAND_TAG%"
goto :eof

:probe_python
if defined PY_CMD goto :eof
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY_CMD=%~1"
goto :eof
