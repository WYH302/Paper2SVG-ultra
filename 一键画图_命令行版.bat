@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   Paper2SVG-ultra v3.0 - 纯净 AutoFigure-Edit 可编辑论文图工具
echo   只抽 Method 章节 + 单风格参考图 -^> 100%% 原生可编辑 SVG
echo ============================================================
echo.

:ASK_PDF
set "PAPER_PDF="
set /p "PAPER_PDF=【第 1 步】请把论文 PDF 文件拖进本窗口按回车: "
set "PAPER_PDF=%PAPER_PDF:"=%"
if "%PAPER_PDF%"=="" goto ASK_PDF
if not exist "%PAPER_PDF%" (
    echo [错误] 找不到文件: %PAPER_PDF%
    echo.
    goto ASK_PDF
)

echo.
set "STYLE_REF="
set /p "STYLE_REF=【第 2 步】请把 1 张顶会风格参考图拖进本窗口按回车（直接回车则零参考图）: "
set "STYLE_REF=%STYLE_REF:"=%"

echo.
set "LANG_OPT=auto"
set /p "LANG_INPUT=【第 3 步】图内文字语言 [直接回车=自动跟随论文 / en=英文 / zh=中文]: "
if /i "%LANG_INPUT%"=="en" set "LANG_OPT=en"
if /i "%LANG_INPUT%"=="zh" set "LANG_OPT=zh"

echo.
echo ------------------------------------------------------------
echo 开始执行 Paper2SVG-ultra v3.0（Method 切片 -^> 底图 -^> 可编辑 SVG）...
echo ------------------------------------------------------------
echo.

if "%STYLE_REF%"=="" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --paper "%PAPER_PDF%" --lang "%LANG_OPT%"
) else (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --paper "%PAPER_PDF%" --style-ref "%STYLE_REF%" --lang "%LANG_OPT%"
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
