@echo off
chcp 65001 > nul
cd /d "%~dp0.."
python scripts\forzy_collector.py
