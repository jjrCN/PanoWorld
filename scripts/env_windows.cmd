@echo off
REM Windows runtime environment for PanoWorld (RTX 50xx / sm_120).
REM Usage:  scripts\env_windows.cmd <command...>
REM Sets up MSVC + CUDA 12.8 (needed for gsplat's JIT kernel build) and runs the command
REM with the repo venv on PATH.
set "PANOWORLD_ROOT=%~dp0.."
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%PANOWORLD_ROOT%\.venv\Scripts;%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "MAX_JOBS=8"
set "DISTUTILS_USE_SDK=1"
REM Keep the gsplat JIT build cache inside the repo so it is shared across shells.
set "TORCH_EXTENSIONS_DIR=%PANOWORLD_ROOT%\.torch_extensions"
set "PYTHONPATH=%PANOWORLD_ROOT%;%PYTHONPATH%"
cd /d "%PANOWORLD_ROOT%"
%*
