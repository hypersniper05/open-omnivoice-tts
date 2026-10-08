@echo off
rem Stop open-omnivoice-tts.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
