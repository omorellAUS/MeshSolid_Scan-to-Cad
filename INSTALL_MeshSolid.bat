@echo off
setlocal
set "D=%~dp0"
if not exist "%D%fusion_addin\install_windows.ps1" set "D=%USERPROFILE%\Downloads\MeshSolid-1.0.2-windows\MeshSolid-1.0.2\MeshSolid-1.0.2\"
if not exist "%D%fusion_addin\install_windows.ps1" set "D=%USERPROFILE%\Downloads\MeshSolid-1.0.2-windows\MeshSolid-1.0.2\"
if not exist "%D%fusion_addin\install_windows.ps1" set "D=%USERPROFILE%\Downloads\MeshSolid-1.0.2\"
if not exist "%D%fusion_addin\install_windows.ps1" (
  echo Could not find the MeshSolid folder. Move this file into the MeshSolid-1.0.2 folder and run it again.
  pause & exit /b 1
)
cd /d "%D%"
echo MeshSolid 1.0.2 installer. Your existing Python 3.10 is NOT changed.
set "PYEXE="
for %%V in (3.13 3.12) do if not defined PYEXE for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do if exist "%%P" set "PYEXE=%%P"
if not defined PYEXE for %%V in (313 312) do if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
if not defined PYEXE (
  echo Python 3.13 not found. Installing it alongside 3.10 ^(it will not replace 3.10 or change PATH^)...
  winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)
if not defined PYEXE (
  echo.
  echo Could not install Python 3.13 automatically.
  echo Install it from https://www.python.org/downloads/windows/ - UNTICK "Add python.exe to PATH" so 3.10 stays your default - then run this file again.
  pause & exit /b 1
)
"%PYEXE%" -c "import sys;sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) else 1)" || (echo Found Python is not 3.12/3.13: %PYEXE% & pause & exit /b 1)
echo Using %PYEXE% for MeshSolid only.
echo This takes a few minutes the first time.
powershell -NoProfile -ExecutionPolicy Bypass -File "%D%fusion_addin\install_windows.ps1" -Python "%PYEXE%" -Gpu
echo.
echo Finished. Take a screenshot of this window and send it.
pause
