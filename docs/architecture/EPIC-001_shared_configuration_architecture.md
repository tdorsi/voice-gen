# EPIC-001 Shared Configuration Architecture

Release: v0.3.0
Epic Branch: `vg_e001_shared_config`
Related Tasks: TASK-011, TASK-012, TASK-013, TASK-014

## Objective

Establish a shared configuration and utility foundation for `voice_gen.py` and `text_to_audio.py` while preserving the current v0.2.0 workflows.

EPIC-001 should remove duplicated UI/logging/prompt code, move local path defaults into configuration, and make voice presets configurable before later hardening and inference enhancements begin.

## Branch Scope

All EPIC-001 implementation work is committed to:

```text
vg_e001_shared_config
```

Commit messages must follow AgentBus policy:

```text
[v0.3.0][vg_e001][TASK-###] Summary
```

## Proposed Modules

### `voice_gen_utils.py`

Shared operational helpers used by both scripts.

Responsibilities:

- ANSI color constants and console formatting.
- Banner, stage header, status, warning, and error output helpers.
- Logging setup with file and console handlers.
- Handler clearing to avoid duplicate log lines when imported or tested repeatedly.
- Interactive prompt helpers.
- Path helper functions that are not specific to one workflow.
- Common exception formatting where useful.

Non-goals:

- No training-stage logic.
- No MOSS inference logic.
- No Voice_Gen pipeline state management.
- No text chunking or audio generation code.

### `voice_gen_config.py`

Configuration loading and validation helpers.

Responsibilities:

- Load `voice_gen.toml` from the repository root by default.
- Merge defaults with user-provided config.
- Normalize Windows path values.
- Validate required path keys before runtime-heavy operations.
- Provide typed access to shared paths and voice presets.
- Keep config errors clear and loggable.

This module keeps parsing/validation out of both application entry points.

### `voice_gen.toml`

Repository-level default configuration.

Initial shape:

```toml
[paths]
moss_root = "D:/AI_Models/Voice/moss-tts"
log_dir = "D:/Development/Voice_Gen/logs"
voices_dir = "D:/AI_Models/Voice/moss-tts/voices"
default_output_dir = "D:/Training_Data/Audio/TestOut"
default_input_file = "D:/Training_Data/Audio/Test_Script/TTS_Script_01.txt"

[moss]
config_dir = "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp"
llama_cpp_dir = "D:/AI_Models/Voice/moss-tts/repo/moss_tts_delay/llama_cpp"
onnx_dir = "D:/AI_Models/Voice/moss-tts/weights/MOSS-Audio-Tokenizer-ONNX"

[voices.lori]
config = "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp/lori.yaml"
reference = "D:/AI_Models/Voice/moss-tts/voices/Lori_ref.wav"

[voices.lilybelle]
config = "D:/AI_Models/Voice/moss-tts/voices/lilybelle.yaml"
reference = "D:/AI_Models/Voice/moss-tts/voices/lilybelle_ref_10s.wav"

[voices.hannah]
config = "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp/hannah.yaml"
reference = "D:/AI_Models/Voice/moss-tts/voices/Hannah_ref.wav"
```

The TOML file is intentionally explicit for v0.3.0. Auto-discovery can supplement it in TASK-014, but source-code hardcoding should be removed first.

## Migration Plan

### TASK-012: Shared Utility Module

1. Create `voice_gen_utils.py`.
2. Move shared visual constants and terminal helpers from both scripts.
3. Move common logging setup into a reusable function.
4. Update `text_to_audio.py` to use the shared helpers first because it has fewer stages.
5. Update `voice_gen.py` to use the shared helpers without changing stage behavior.
6. Run syntax checks on all touched Python files.

Review checkpoint: Claude reviews that the shared helper API is small, stable, and does not mix training/inference responsibilities.

### TASK-013: Shared Configuration System

1. Create `voice_gen.toml`.
2. Create `voice_gen_config.py`.
3. Add config loading with path normalization and validation.
4. Wire shared paths into `text_to_audio.py`.
5. Wire shared paths into `voice_gen.py` where safe.
6. Document config keys in README.

Review checkpoint: Claude reviews config layout, validation behavior, and compatibility with current local paths.

### TASK-014: Voice Presets and Defaults

1. Move Lori, Lilybelle, and Hannah presets out of `text_to_audio.py`.
2. Load presets from `voice_gen.toml`.
3. Preserve `--voice all` behavior.
4. Use configured interactive defaults for input and output prompts.
5. Keep timestamped output collision behavior unchanged.
6. Document adding a new voice preset.

Review checkpoint: Claude reviews removal of hardcoded presets/default paths and verifies workflows still resolve existing voices.

## Compatibility Requirements

Existing v0.2.0 commands should continue to work:

```bat
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --dry-run
voice_gen.bat --voice MyVoice --input D:\Audio\raw --output D:\Audio\output
```

No EPIC-001 change should require public APIs, cloud dependencies, or new package installs beyond Python standard library support already available in the active runtime.

TOML parsing should use `tomllib` on Python 3.11+. If compatibility below Python 3.11 is required later, introduce a deliberate fallback plan rather than quietly adding a dependency.

## Risks

- `voice_gen.py` has many stage-specific log and status calls; migration should avoid changing stage order or state-file semantics.
- `text_to_audio.py` currently handles Windows/WSL path normalization for Lilybelle. That behavior must survive config migration.
- Config-driven paths can fail later than import-time constants if validation is weak. Validate early and log clearly.
- Shared helpers can become too broad. Keep workflow-specific logic in the scripts.

## Proposed Review Artifacts

Codex should submit the following to Claude for TASK-011 review:

- This architecture note.
- A short diff summary of any scaffolding added before implementation.
- The planned order of TASK-012, TASK-013, and TASK-014.

Claude should review for:

- Separation of concerns.
- Config shape clarity.
- Migration safety.
- Compatibility with EPIC-001 acceptance criteria.
