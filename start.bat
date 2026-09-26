@echo off
REM Double-click to start Triffy on Windows. Everything it does is in start.py.
cd /d "%~dp0"
set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set PY=py -3
)
if not defined PY set PY=python
%PY% start.py
if errorlevel 1 (
  echo.
  echo Triffy did not start. If Python is missing, install it from
  echo https://www.python.org/downloads/ and tick "Add python.exe to PATH".
)
pause
