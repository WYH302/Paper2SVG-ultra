@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   Paper2SVG-ultra - 顶会论文总体框架图可编辑 SVG 工作台
echo ============================================================
echo.
echo 正在启动本地服务 (http://127.0.0.1:8090) ...
echo 提示：使用时请不要关闭此黑色窗口，用完直接关掉即可。
echo.

:: 延迟 2 秒自动打开浏览器
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8090"

"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --serve --host 127.0.0.1 --port 8090

echo.
echo 服务已停止。
pause
