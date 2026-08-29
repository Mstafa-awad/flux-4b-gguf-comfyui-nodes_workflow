@echo off
setlocal EnableExtensions
title Install FLUX2 Klein GGUF Staged Nodes

set "PACKAGE_DIR=%~dp0"
set "COMFY_ROOT=%~1"
set "TARGET_NAME=flux-4b-gguf-comfyui-nodes_workflow"

if not defined COMFY_ROOT (
  echo Drag your ComfyUI folder onto this BAT, or paste the path below.
  set /p "COMFY_ROOT=ComfyUI folder: "
)

for %%I in ("%COMFY_ROOT%") do set "COMFY_ROOT=%%~fI"
if not exist "%COMFY_ROOT%\main.py" (
  echo.
  echo ERROR: main.py was not found in:
  echo %COMFY_ROOT%
  echo Select the actual ComfyUI folder, not only ComfyUI-Easy-Install.
  pause
  exit /b 1
)

if not exist "%PACKAGE_DIR%nodes.py" (
  echo ERROR: nodes.py was not found beside this installer.
  pause
  exit /b 1
)

set "TARGET_DIR=%COMFY_ROOT%\custom_nodes\%TARGET_NAME%"
for %%I in ("%PACKAGE_DIR%.") do set "PACKAGE_FULL=%%~fI"
for %%I in ("%TARGET_DIR%") do set "TARGET_FULL=%%~fI"

if /I "%PACKAGE_FULL%"=="%TARGET_FULL%" goto :dependency

echo.
echo Installing staged FLUX2 nodes...
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
xcopy "%PACKAGE_DIR%*" "%TARGET_DIR%\" /E /I /Y /EXCLUDE:"%PACKAGE_DIR%.install-exclude.txt" >nul
if errorlevel 1 goto :copy_error

:dependency
set "GGUF_DIR=%COMFY_ROOT%\custom_nodes\ComfyUI-GGUF"
if not exist "%GGUF_DIR%\nodes.py" (
  where git >nul 2>nul
  if errorlevel 1 (
    echo.
    echo ERROR: ComfyUI-GGUF is missing and Git was not found.
    echo Install ComfyUI-GGUF with ComfyUI Manager, then run this installer again.
    pause
    exit /b 1
  )
  echo Installing ComfyUI-GGUF...
  git clone https://github.com/city96/ComfyUI-GGUF.git "%GGUF_DIR%"
  if errorlevel 1 goto :gguf_error
) else (
  echo ComfyUI-GGUF is already installed.
)

set "PYTHON_EXE="
if exist "%COMFY_ROOT%\..\python_embeded\python.exe" set "PYTHON_EXE=%COMFY_ROOT%\..\python_embeded\python.exe"
if not defined PYTHON_EXE if exist "%COMFY_ROOT%\python_embeded\python.exe" set "PYTHON_EXE=%COMFY_ROOT%\python_embeded\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if defined PYTHON_EXE if exist "%GGUF_DIR%\requirements.txt" (
  echo Installing ComfyUI-GGUF Python requirements...
  "%PYTHON_EXE%" -s -m pip install -r "%GGUF_DIR%\requirements.txt"
  if errorlevel 1 goto :pip_error
)

set "WF_DIR=%COMFY_ROOT%\user\default\workflows\Flux2-Klein-GGUF-Staged"
if not exist "%WF_DIR%" mkdir "%WF_DIR%"
xcopy "%PACKAGE_DIR%workflows\*.json" "%WF_DIR%\" /Y >nul

echo.
echo DONE. Restart ComfyUI, then load a workflow from:
echo %WF_DIR%
echo.
echo The installer does not download model weights. See README.md for links.
pause
exit /b 0

:copy_error
echo ERROR: Could not copy the custom-node package.
pause
exit /b 1

:gguf_error
echo ERROR: Could not clone ComfyUI-GGUF.
pause
exit /b 1

:pip_error
echo ERROR: Could not install ComfyUI-GGUF requirements.
pause
exit /b 1
