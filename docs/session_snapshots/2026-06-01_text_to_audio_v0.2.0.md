# Session Snapshot: Text-to-Audio Utility v0.2.0

Date: 2026-06-01
Branch: `voice-gen_0.2.0`

## Summary

Added a text-to-audio inference utility for Voice_Gen that converts `.txt` files to WAV output using the local MOSS-TTS llama.cpp pipeline. The utility is intended to feel like the existing `voice_gen.py` training pipeline while serving a different workflow: local text inference instead of voice preparation and fine-tuning.

## Files Added

- `text_to_audio.py` - local MOSS text-to-WAV converter.
- `text_to_audio.bat` - Windows launcher that activates the `moss-tts` conda environment.
- `docs/session_snapshots/2026-06-01_text_to_audio_v0.2.0.md` - this snapshot.

## Files Updated

- `README.md` - updated to v0.2.0 with text-to-audio usage, logging, output naming, and troubleshooting.
- `.gitignore` - ignores generated `output/` pipeline artifacts.

## Implemented Behavior

- Supports built-in voice presets: `lori`, `lilybelle`, `hannah`, and `all`.
- Reads text files, chunks content, generates audio per chunk, and concatenates chunks into one WAV.
- Uses local MOSS assets only; no cloud or public API path is used.
- Normalizes WSL-style `/mnt/d/...` paths in Lilybelle config at runtime for Windows execution.
- Provides dry-run chunk inspection without loading MOSS or generating audio.
- Logs each run under `D:\Development\Voice_Gen\logs`.
- If output exists and overwrite is declined, writes a timestamped output such as `TTS_Script_01_hannah_075924.wav`.
- Uses a Voice_Gen-style terminal interface with banner, stage sections, status lines, and completion summary.

## Validation Performed

- `python -m py_compile text_to_audio.py`
- Dry-run chunk validation for `TTS_Script_01.txt`.
- Runtime Hannah generation completed successfully to timestamped output:
  - `D:\Training_Data\Audio\Test_Script\testout\TTS_Script_01_hannah_075924.wav`
  - Log: `D:\Development\Voice_Gen\logs\20260601_075924_text_to_audio_hannah.log`

## Known Follow-Ups

- Revisit the interactive interface and overall UI so `voice_gen.py` and `text_to_audio.py` share common helpers instead of duplicating banner, logging, prompt, and stage code.
- Normalize default output directory prompts around the preferred test output folder: `D:\Training_Data\Audio\TestOut`.
- Consider making voice presets configurable instead of hardcoded.
- Consider preserving per-chunk intermediate WAVs optionally for failed-run recovery and QA.
