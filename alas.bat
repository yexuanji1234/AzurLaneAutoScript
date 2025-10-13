@echo off
chcp 65001 >nul  REM 使用 UTF-8 编码，确保中文路径和输出不乱码

REM 1. 切换到 ALAS 文件夹（自己更改文件路径）
echo 正在切换到 ALAS 文件夹...
cd /d "C:\Users\examp\Desktop\ALAS"

REM 2. 启动 Python GUI
echo 正在启动 ALAS，并激活 Conda 环境运行 gui.py...
REM 使用 PowerShell 启动 Conda 环境并运行脚本
start "" powershell -NoExit -Command "conda activate alas; python gui.py"

REM 3. 同时打开指定的网址
echo 正在打开网址 http://127.0.0.1:22267 ...
start http://127.0.0.1:22267

exit /b

