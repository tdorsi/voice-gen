# Changelog

All notable changes to Voice_Gen are documented here.

---

## [v0.3.0] — 2026-06-21

### Added

- Shared configuration framework:
  - `voice_gen.toml` centralizes runtime paths, defaults, and voice presets.
  - `voice_gen_config.py` loads and validates configuration relative to the project.
  - `voice_gen_utils.py` provides shared terminal UI, logging, prompt, and output helpers.
  - Voice presets and default input/output paths are configuration-driven and discoverable.
- Voice training hardening:
  - Fresh runs protect existing output directories by default; `--force` provides an
    explicit, logged override while `--from-stage` continues to support safe resume.
  - `--dry-run` performs input preparation through reference selection, then stops before
    transcription, downloads, token encoding, fine-tuning, sample generation, and export.
  - `--log-file` redirects the run log and creates missing parent directories.
  - Graceful `KeyboardInterrupt` handling exits cleanly with status 130.
  - Dependency failures are recorded in the run log before exit.
- Text-to-audio enhancements:
  - `--keep-chunks` preserves numbered per-chunk WAV files while retaining the final
    concatenated output.
  - Live chunk progress reporting includes chunk counts and character counts.
  - ETA reporting uses measured completed-chunk throughput and accounts for remaining
    voices when `--voice all` is selected.

### Changed

- `voice_gen.py` and `text_to_audio.py` now share configuration, logging, and terminal
  behavior instead of maintaining separate hardcoded defaults.
- README usage and configuration guidance now covers the v0.3.0 training and inference
  workflows.

---

## [v0.2.0] — 2026-06-01

### Added

- `text_to_audio.py` — inference-only utility that converts `.txt` files to WAV audio
  using the local MOSS-TTS llama.cpp pipeline. Supports `lori`, `lilybelle`, `hannah`,
  and `all` voice presets, configurable chunk sizing, silence padding, dry-run mode,
  overwrite protection with timestamped fallback, and recursive context-overflow retry.
- `text_to_audio.bat` — Windows launcher: activates the `moss-tts` conda environment
  and forwards all arguments to `text_to_audio.py`.
- `ONNX_DIR` constant (`MOSS_ROOT/weights/MOSS-Audio-Tokenizer-ONNX`) added to the
  module-level path block for use by the dependency checker.
- `check_dependencies()` — pre-flight guard that runs before interactive prompts.
  Verifies `numpy` and `soundfile` are importable, confirms `ggml.dll` and `llama.dll`
  are present in `LLAMA_CPP_DIR`, and confirms `encoder.onnx` and `decoder.onnx` exist
  in `ONNX_DIR`. Fails fast with actionable error messages rather than mid-run crashes.
- `validate_args()` — argument bounds check that runs after interactive prompts but
  before logging is set up. Guards `--chunk-chars` (>= 10), `--max-new-tokens` (>= 1),
  and `--silence-ms` (>= 0) against nonsensical values.

### Changed

- Terminal symbols aligned with `voice_gen.py` so both tools feel like one suite:
  - OK lines: `OK` → `✓`
  - Error lines: `X` → `✗`
  - Banner and voice-loop separators: `=` → `═` (U+2550, box-drawing double horizontal)
  - Stage header separators: `-` → `─` (U+2500, box-drawing light horizontal)
  - Log file separators updated to match (`=` → `═`).
- `text_to_audio.bat` header comment expanded to include four usage examples,
  matching the documentation style of `voice_gen.bat`.

---

## [v0.1.0] — 2026-05-31

### Added

- `voice_gen.py` — ten-stage MOSS-TTS voice cloning and fine-tuning pipeline:
  scan, split, clean, score/elect reference, transcribe (faster-whisper), download
  HF weights, encode audio tokens, QLoRA fine-tune, generate sample outputs,
  write voice YAML config.
- `voice_gen.bat` — Windows launcher for the pipeline.
- `train_qlora.py` — single-GPU QLoRA 4-bit fine-tuning script.
- `merge_and_convert_lora.py` — LoRA adapter merge and GGUF conversion helper.
- Stage-level state persistence via `.voice_gen_state.json` enabling `--from-stage`
  resume on failure.
- Dual-handler logging (file=DEBUG, console=INFO) to `D:\Development\Voice_Gen\logs\`.
