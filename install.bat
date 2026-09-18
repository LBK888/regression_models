@echo off
setlocal enabledelayedexpansion
title Regression Models - Environment Installer
cd /d "%~dp0"

REM ===================== configurable =====================
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"
REM Pin a version here for reproducibility, e.g. set "TORCH_SPEC=torch==2.11.0"
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

REM --------------------------------------------------- [4/6] requirements
echo [4/6] Installing project dependencies from %REQ_FILE%...
if not exist "%REQ_FILE%" goto :err_no_req
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :err_req
echo.

REM -------------------------------------------- [5/6] CUDA detect + torch
echo [5/6] Detecting CUDA and installing PyTorch...
set "CUDA_VER="
set "CU_MAJ=0"
set "CU_MIN=0"

where nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_cpu

for /f "tokens=3 delims=:" %%a in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do (
    for /f "tokens=1 delims= " %%b in ("%%a") do set "CUDA_VER=%%b"
)
if not defined CUDA_VER goto :torch_cpu

for /f "tokens=1,2 delims=." %%m in ("!CUDA_VER!") do (
    set "CU_MAJ=%%m"
    set "CU_MIN=%%n"
)
if not defined CU_MIN set "CU_MIN=0"
echo       NVIDIA driver reports CUDA !CUDA_VER!

if !CU_MAJ! GEQ 13 goto :torch_cu130
if !CU_MAJ! EQU 12 goto :torch_cuda12
if !CU_MAJ! EQU 11 goto :torch_cu118
goto :torch_cpu

:torch_cuda12
if !CU_MIN! GEQ 8 goto :torch_cu128
goto :torch_cu126

:torch_cu130
set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
set "TORCH_LABEL=CUDA 13.0 build"
goto :torch_install

:torch_cu128
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_LABEL=CUDA 12.8 build"
goto :torch_install

:torch_cu126
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
set "TORCH_LABEL=CUDA 12.6 build"
goto :torch_install

:torch_cu118
set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
set "TORCH_LABEL=CUDA 11.8 build"
goto :torch_install

:torch_cpu
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU-only build"
echo       No usable NVIDIA CUDA runtime detected.

:torch_install
echo       Installing PyTorch: %TORCH_LABEL%
echo       Index: %TORCH_INDEX%
"%VENV_PY%" -m pip install %TORCH_SPEC% --index-url %TORCH_INDEX%
if errorlevel 1 goto :err_torch
echo.

REM ------------------------------------------------- [6/6] verify + folders
echo [6/6] Verifying the installation...
"%VENV_PY%" -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib, openpyxl, PyQt6.QtWidgets, torch; print('  numpy        ', numpy.__version__); print('  pandas       ', pandas.__version__); print('  scikit-learn ', sklearn.__version__); print('  matplotlib   ', matplotlib.__version__); print('  PyQt6        ', 'ok'); print('  torch        ', torch.__version__); print('  CUDA available', torch.cuda.is_available()); print('  GPU          ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
if errorlevel 1 goto :err_verify

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
echo   Optional model adapters (CatBoost, XGBoost, LightGBM, TabM, RealMLP)
echo   are commented out in requirements.txt. Uncomment a line and run this
echo   installer again to add that model family to the training tab.
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
echo         Tip: brand-new Python releases often have no wheels yet - set
echo              PY_OVERRIDE at the top of install.bat to an older 3.1x and retry.
pause
exit /b 1

:err_torch
echo.
echo [ERROR] PyTorch installation failed (%TORCH_LABEL%).
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
:probe_python
if defined PY_CMD goto :eof
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY_CMD=%~1"
goto :eof
