@echo off
setlocal
cd /d "%~dp0"
if not exist audit-back\.venv\Scripts\python.exe (
  echo Prepare o ambiente seguindo docs/audit/README.md.
  exit /b 1
)
audit-back\.venv\Scripts\python.exe audit-back\run.py serve
