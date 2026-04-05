@echo off
:: Voice_Gen launcher — activates moss-tts conda env and runs the pipeline
::
:: Usage:
::   voice_gen.bat
::   voice_gen.bat --voice MyVoice --input D:\Audio\raw --output D:\Audio\out
::   voice_gen.bat --from-stage 5
::   voice_gen.bat --skip-finetune

call conda activate moss-tts 2>nul
if errorlevel 1 (
    echo ERROR: Could not activate moss-tts conda environment.
    echo Run:  conda activate moss-tts
    pause
    exit /b 1
)

python "%~dp0voice_gen.py" %*
if errorlevel 1 (
    echo.
    echo Pipeline exited with an error. Check output above.
    pause
)
