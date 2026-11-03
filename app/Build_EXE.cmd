@echo off
setlocal
cd /d "%~dp0"
rem Ayna (أين سُجِّل؟) desktop app: one-file EXE, no torch (detector runs through ONNX Runtime).
rem Needs: pip install pyqt5 pyinstaller onnxruntime numpy scipy pandas matplotlib soundfile requests pillow
rem        plus dopplertrack and overpass installed (pip install -e ..\..\..\systmatic review\dopplertrack  and  pip install -e ..)
set "PY=python"
if not exist models mkdir models
if not exist models\ridgenet.onnx echo WARNING: models\ridgenet.onnx missing - the app will use the classical detector only.
if not exist bin\ffmpeg.exe (
  echo copying ffmpeg from imageio-ffmpeg ...
  "%PY%" -c "import imageio_ffmpeg, shutil, os; os.makedirs('bin', exist_ok=True); shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), 'bin/ffmpeg.exe')" || goto :fail
)
set "OPTS=--noconfirm --clean --onefile --windowed --name Ayna --add-binary bin\ffmpeg.exe;bin --add-data models;models --hidden-import overpass --hidden-import dopplertrack --exclude-module torch --exclude-module torchvision --exclude-module PySide6 --exclude-module PyQt6 --exclude-module tkinter --exclude-module PyQt5.QtWebEngineWidgets --exclude-module IPython --exclude-module jupyter"
"%PY%" -m PyInstaller %OPTS% ayna.py || goto :fail
echo.
echo Done: dist\Ayna.exe
explorer dist
exit /b 0
:fail
echo BUILD FAILED
pause
exit /b 1
