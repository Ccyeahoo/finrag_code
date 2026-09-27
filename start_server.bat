@echo off
REM ============================================================================
REM  finrag -- start the Q&A server (stage 5 needs a key to exercise the LLM path)
REM
REM  This file stores NO secret. It asks for the key at run time and puts it
REM  only into this window's environment. Nothing is written to disk.
REM  (ASCII only on purpose -- a .bat with Chinese in it shows up as mojibake
REM   in cmd.exe, which would make a confusing step worse.)
REM ============================================================================
cd /d "%~dp0"

set "FINRAG_BASE_URL=https://api.deepseek.com/v1"
REM  Model name is left unset on purpose: GET /v1/models on this endpoint does
REM  not list "deepseek-chat", so the code picks one from the list by itself.

if not "%FINRAG_API_KEY%"=="" goto havekey
echo.
echo Paste your API key below, then press Enter.
echo (Right-click pastes in this window. The key is NOT saved anywhere.)
echo.
set /p FINRAG_API_KEY=Key:

:havekey
if "%FINRAG_API_KEY%"=="" (
  echo.
  echo No key entered. Nothing started.
  echo.
  pause
  exit /b 1
)

echo.
echo Base URL : %FINRAG_BASE_URL%
echo Key      : set (deliberately not shown)
echo.
echo Loading the index, then about 30 seconds of warm-up.
echo Wait until you see:  started http://127.0.0.1:8000/
echo Keep this window open -- closing it stops the server.
echo.

REM  Pick a python: %FINRAG_PY% wins, then the known local install, then PATH.
REM  Bare "python" on Windows is often the Microsoft Store stub, which does
REM  nothing useful -- so try the real path first.
set "PYEXE=%FINRAG_PY%"
if not "%PYEXE%"=="" goto havepy
set "PYEXE=C:\Users\yexinyu\AppData\Local\Python\bin\python.exe"
if exist "%PYEXE%" goto havepy
set "PYEXE=python"

:havepy
echo Python   : %PYEXE%
"%PYEXE%" -X utf8 -u src\serve.py --port 8000

echo.
echo ---- server exited ----
pause
