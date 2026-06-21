@echo off
:: Voice_Gen — text-to-audio conversion with local MOSS-TTS voices.
::
:: Usage:
::   text_to_audio.bat
::   text_to_audio.bat --input D:\Scripts\my_text.txt --voice lori
::   text_to_audio.bat --input D:\Scripts\my_text.txt --voice all
::   text_to_audio.bat --input D:\Scripts\my_text.txt --dry-run --show-chunks

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
