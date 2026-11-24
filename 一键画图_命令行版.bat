@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   Paper2SVG-ultra - 一键生成顶会论文总体框架图 (Editable SVG)
echo   读取 .env 中的 OpenAI Responses API 模型配置
echo ============================================================
echo.

:ASK_PDF
set "PAPER_PDF="
set /p "PAPER_PDF=【第 1 步】请把你的论文 PDF 文件直接拖进本窗口，然后按回车: "
set "PAPER_PDF=%PAPER_PDF:"=%"
if "%PAPER_PDF%"=="" goto ASK_PDF
if not exist "%PAPER_PDF%" (
    echo [错误] 找不到文件: %PAPER_PDF%
    echo 请重新拖入有效的 PDF 文件！
    echo.
    goto ASK_PDF
)

echo.
set "REFS_DIR="
set /p "REFS_DIR=【第 2 步】请把参考图文件夹拖进本窗口按回车（直接回车则使用默认顶会风格）: "
set "REFS_DIR=%REFS_DIR:"=%"

echo.
set "LANG_OPT=auto"
set /p "LANG_INPUT=【第 3 步】图内文字语言 [直接回车=自动跟随论文 / en=英文 / zh=中文]: "
if /i "%LANG_INPUT%"=="en" set "LANG_OPT=en"
if /i "%LANG_INPUT%"=="zh" set "LANG_OPT=zh"

echo.
echo ------------------------------------------------------------
echo 开始执行 6 阶段顶配出图（全程约 5~8 分钟，请耐心等待）...
echo ------------------------------------------------------------
echo.

if "%REFS_DIR%"=="" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --paper "%PAPER_PDF%" --lang "%LANG_OPT%"
) else (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --paper "%PAPER_PDF%" --refs "%REFS_DIR%" --lang "%LANG_OPT%"
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   生成完成！正在自动打开输出目录 outputs\ ...
    echo   最终矢量图: final.svg  ^|  高清预览: final.png
    echo ============================================================
    explorer "%~dp0outputs"
) else (
    echo.
    echo [失败] 运行过程中出现错误，请查看上方日志信息。
)

echo.
pause
