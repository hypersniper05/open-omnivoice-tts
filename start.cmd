@echo off
rem Start open-omnivoice-tts (double-click or run from cmd). Works without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
