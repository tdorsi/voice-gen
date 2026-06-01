@echo off
:: Convert text files to WAV audio with local MOSS-TTS voices.

call conda activate moss-tts 2>nul
if errorlevel 1 (
    echo ERROR: Could not activate moss-tts conda environment.
    echo Run:  conda activate moss-tts
    pause
    exit /b 1
)

python "%~dp0text_to_audio.py" %*
if errorlevel 1 (
    echo.
    echo Text-to-audio conversion exited with an error. Check output above.
    pause
)
