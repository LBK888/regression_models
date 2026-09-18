@echo off
setlocal
title Regression Models
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "APP=main.py"

echo ==========================================
echo   Regression Models - Startup
echo ==========================================
echo.

REM ------------------------------------------ [1/4] virtual environment
echo [1/4] Checking virtual environment...
if not exist "%VENV_PY%" goto :err_no_venv
echo       OK - %VENV_DIR%
echo.

REM ------------------------------------------------ [2/4] program files
echo [2/4] Checking program files...
if not exist "%APP%" goto :err_no_app
if not exist "regression_core.py" goto :err_no_app
if not exist "regression_v4\ui.py" goto :err_no_app
if not exist "app\main_window.py" goto :err_no_app
echo       OK - all modules present
echo.

REM ------------------------------------------------- [3/4] dependencies
echo [3/4] Checking dependencies...
"%VENV_PY%" -c "import numpy, pandas, sklearn, matplotlib, joblib, openpyxl, torch, PyQt6.QtWidgets" >nul 2>&1
if errorlevel 1 goto :err_deps
"%VENV_PY%" -c "import torch; print('       torch', torch.__version__, '|', ('GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU mode (no CUDA device)')"
"%VENV_PY%" %APP% --self-test >nul 2>&1
if errorlevel 1 goto :err_selftest
echo       OK - dependencies and engine self-test passed
echo.

REM ------------------------------------------------------- [4/4] launch
echo [4/4] Starting the application...
echo.
"%VENV_PY%" "%APP%"
if errorlevel 1 goto :err_runtime
exit /b 0

REM ============================== errors ==============================
:err_no_venv
echo       [ERROR] No virtual environment found at "%VENV_DIR%".
echo.
echo       Run install.bat first - it creates the environment and installs
echo       every dependency including the right PyTorch build.
echo.
pause
exit /b 1

:err_no_app
echo       [ERROR] A required program file is missing next to start.bat.
echo               Expected main.py, regression_core.py, regression_v4\ and app\.
echo               Make sure start.bat sits in the project folder.
echo.
pause
exit /b 1

:err_deps
echo       [ERROR] The virtual environment is missing or has broken packages.
echo               Details:
"%VENV_PY%" -c "import numpy, pandas, sklearn, matplotlib, joblib, openpyxl, torch, PyQt6.QtWidgets"
echo.
echo       Run install.bat to repair the environment.
echo.
pause
exit /b 1

:err_selftest
echo       [ERROR] The engine self-test failed. Details:
"%VENV_PY%" %APP% --self-test
echo.
echo       Run install.bat to repair the environment.
echo.
pause
exit /b 1

:err_runtime
echo.
echo [ERROR] An exception occurred while running %APP%. See the messages above.
pause
exit /b 1
