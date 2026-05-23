@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m chess_project.io.uci --model model.pt %*
