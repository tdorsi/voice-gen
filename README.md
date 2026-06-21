# Voice_Gen
## Version
Current Version: v0.2.0

A fully local voice cloning pipeline that fine-tunes large text-to-speech models on consumer GPUs — no cloud APIs, no subscriptions, no data leaving your machine.

---

## What This Is

Voice_Gen is an automated 10-stage pipeline that takes raw audio recordings and produces a fine-tuned TTS model ready for local inference using MOSS-TTS.

Built and tested on:
- NVIDIA RTX 5070 (12GB VRAM)
- Fully offline environment

Voice_Gen is designed as a reusable voice creation tool, not a one-off experiment.

---

## Why I Built This

Most voice cloning solutions either:
- depend on cloud APIs, or  
- require high-end hardware  

I wanted a system that:
- runs entirely locally  
- works on consumer GPUs  
- can be reused to create multiple character voices  

This project is also a foundational component of a larger system (Nova-Vex), where these voices are used in real-time AI-driven character interactions.

---

## Key Capabilities

- Fine-tunes an 8B parameter TTS model on a 12GB GPU  
- Fully automated pipeline from raw audio → trained voice  
- Converts text files to WAV audio using local MOSS-TTS voice configs  
- Runs completely offline (no external APIs)  
- Produces reusable voice models for downstream systems  
- Resume-safe pipeline with stage-level recovery  

---

## Pipeline Overview

Voice_Gen runs a 10-stage pipeline that handles everything from raw audio to a deployable voice config:

| Stage | Description |
|-------|-------------|
| 1 | Scan input directory, classify files by duration |
| 2 | Split long files (>15s) into 10–15s clips at silence boundaries |
| 3 | Noise-reduce (`afftdn`) and normalise to 24 kHz mono WAV |
| 4 | Score clips by quality; elect best as `reference.wav` |
| 5 | Transcribe with faster-whisper → training JSONL |
| 6 | Verify / download HuggingFace model weights |
| 7 | Encode audio token codes (`prepare_data.py`) |
| 8 | QLoRA fine-tune (4-bit NF4 + LoRA adapter, single GPU) |
| 9 | Generate 5 test WAV samples from the checkpoint |
| 10 | Write voice YAML config + copy reference to server voices dir |

State is saved after every stage. If a run fails, resume from where it stopped with:

```bash
--from-stage N
```

---

## Requirements

### Hardware
- NVIDIA GPU with 12+ GB VRAM (tested on RTX 5070 12 GB)
- 16+ GB system RAM recommended

### Software
- Windows 10/11 (Linux untested but likely works with path adjustments)
- Miniconda or Anaconda
- MOSS-TTS repository cloned locally

### Python Environment

```bash
conda create -n moss-tts python=3.11
conda activate moss-tts

pip install torch==2.9.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate peft bitsandbytes
pip install faster-whisper soundfile huggingface_hub rich
pip install pyyaml
```

> Do NOT install `torchcodec` — it causes DLL conflicts with torchaudio in this environment.

## ffmpeg

Voice_Gen requires ffmpeg for audio processing (splitting, normalization, noise reduction).

⚠️ ffmpeg is NOT included in this repository.

### Installation (Windows)

Download a static build from:
https://github.com/BtbN/FFmpeg-Builds/releases

Extract and either:
- Add ffmpeg to your system PATH  
- OR place it in a known location and update your script accordingly  

Voice_Gen expects ffmpeg to be available on the system.

---

## Usage

## Configuration

Voice_Gen loads shared settings from `voice_gen.toml` in the repository root. The config path is resolved relative to the Python module location, so `voice_gen.py` and `text_to_audio.py` can be launched from batch files or other working directories.

Key sections:

| Section | Keys |
|---------|------|
| `[paths]` | `moss_root`, `moss_repo`, `weights_dir`, `log_dir`, `voices_dir`, `default_output_dir`, `default_input_file`, `ffmpeg_dir` |
| `[moss]` | `config_dir`, `llama_cpp_dir`, `onnx_dir` |
| `[text_to_audio]` | `default_voice` |
| `[voices.<name>]` | `config`, `reference` |

If `voice_gen.toml` is missing or invalid, the tools fail fast with a clear config error. If required runtime paths are missing, the run log records each missing configured path before the tool exits.

### Voice Training Pipeline

Use `voice_gen.bat` when preparing, training, and exporting a reusable voice.

```bat
voice_gen.bat --voice MyVoice --input D:\Audio\raw --output D:\Audio\output
```

### Common Workflows

```bat
# Full run from scratch
voice_gen.bat --voice <VoiceName> --input D:\Training_Data\Audio\<VoiceName> --output D:\Voices\<VoiceName>

# Resume from fine-tuning
voice_gen.bat --from-stage 8

# Intentionally reuse an existing output directory for a fresh run
voice_gen.bat --voice <VoiceName> --input D:\Training_Data\Audio\<VoiceName> --output D:\Voices\<VoiceName> --force

# Write the run log to a specific path
voice_gen.bat --voice <VoiceName> --input D:\Training_Data\Audio\<VoiceName> --output D:\Voices\<VoiceName> --log-file D:\Logs\<VoiceName>.log

# Plan input prep without transcription or training
voice_gen.bat --voice <VoiceName> --input D:\Training_Data\Audio\<VoiceName> --output D:\Voices\<VoiceName> --dry-run

# Zero-shot only (no fine-tuning)
voice_gen.bat --skip-finetune
```

Fresh Voice_Gen runs are non-destructive by default. If the selected output directory already exists, the tool stops before writing training artifacts. Use `--from-stage N` to resume an existing run. Use `--force` only when you intentionally want a fresh run to reuse an existing output directory; the override is written to the run log.
Use `--dry-run` to run input scanning, splitting, cleanup, scoring, and reference selection, then stop before transcription, weight checks/downloads, token encoding, fine-tuning, sample generation, or config export.

### Text-to-Audio Conversion

Use `text_to_audio.bat` when converting an existing `.txt` file to one or more WAV files with local MOSS-TTS voices. This is inference only; it does not train or fine-tune.

```bat
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --output D:\Training_Data\Audio\TestOut
```

Configured voices:

| Voice | Config | Reference |
|-------|--------|-----------|
| `lori` | `D:\AI_Models\Voice\moss-tts\repo\configs\llama_cpp\lori.yaml` | `D:\AI_Models\Voice\moss-tts\voices\Lori_ref.wav` |
| `lilybelle` | `D:\AI_Models\Voice\moss-tts\voices\lilybelle.yaml` | `D:\AI_Models\Voice\moss-tts\voices\lilybelle_ref_10s.wav` |
| `hannah` | `D:\AI_Models\Voice\moss-tts\repo\configs\llama_cpp\hannah.yaml` | `D:\AI_Models\Voice\moss-tts\voices\Hannah_ref.wav` |
| `all` | Runs every configured voice sequentially | Per voice |

Voice choices are discovered from the `[voices.<name>]` sections in `voice_gen.toml`. To add a voice, add its config and reference paths, then use the section name with `--voice`:

```toml
[voices.myvoice]
config = "D:/AI_Models/Voice/moss-tts/voices/myvoice.yaml"
reference = "D:/AI_Models/Voice/moss-tts/voices/myvoice_ref.wav"
```

Set `[text_to_audio] default_voice` to change the voice selected when `--voice` is omitted. `--voice all` runs every configured voice in file order.

Common text-to-audio workflows:

```bat
# Prompt interactively
text_to_audio.bat

# Generate one voice
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --output D:\Training_Data\Audio\TestOut

# Generate all built-in voices
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice all --output D:\Training_Data\Audio\TestOut

# Replace an existing output file
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --output D:\Training_Data\Audio\TestOut --overwrite

# Dry-run chunking without loading MOSS or generating audio
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --dry-run

# Preserve intermediate numbered WAV files for each chunk
text_to_audio.bat --input script.txt --voice hannah --keep-chunks
```

If the target output already exists and overwrite is not selected, `text_to_audio.py` preserves the existing file and writes a timestamped sibling:

```
TTS_Script_01_hannah.wav
TTS_Script_01_hannah_075924.wav
```

### Progress and ETA

For long-running inference tasks (especially with `--voice all`), the tool reports real-time progress and an Estimated Time of Arrival (ETA).

```
[3] Generating audio
    Processing chunk 1 of 5 (180 chars) | ETA: estimating...
    0.45s audio generated in 2.12s
    Processing chunk 2 of 5 (175 chars) | ETA: 0m 10s
```

The ETA factors in all remaining characters across the entire run, providing a stable estimate based on the characters-per-second (CPS) throughput of completed chunks.

For long or difficult text, reduce chunk size and generation length:

```bat
text_to_audio.bat --input D:\Training_Data\Audio\Test_Script\TTS_Script_01.txt --voice hannah --chunk-chars 100 --max-new-tokens 600
```

Stop any running MOSS inference server before large text-to-audio runs if VRAM is constrained. A loaded server can consume most of a 12GB GPU.

---

## Audio Input Guidelines

| Recommendation | Detail |
|----------------|--------|
| Minimum audio | ~1–2 minutes total |
| Better quality | 5–10 minutes across recordings |
| Format | WAV, MP3, FLAC, M4A, OGG, AAC, OPUS |
| Content | Clean speech, single speaker |
| Clip handling | Auto-split; clips under 8s skipped |

More audio = better voice quality.

---

## Output Structure

```
output/
  clips/
  reference.wav
  train_raw.jsonl
  train_with_codes.jsonl
  checkpoint/
    adapter-final/
  samples/
  <VoiceName>.yaml
```

---

## QLoRA Fine-Tuning

Stage 8 uses a custom QLoRA trainer to fit an 8B model into a 12GB GPU:

| Technique | Effect |
|-----------|--------|
| 4-bit NF4 quantization | Reduces base model memory (~4 GB vs ~16 GB) |
| LoRA adapter | Only 0.86% of parameters trained |
| Gradient checkpointing | Reduces activation memory |
| bf16 autocast | Efficient forward pass |

Peak VRAM during training: ~10.8 GB.

---

## Logs

Each training or text-to-audio run generates a timestamped log:

```
logs/<YYYYMMDD_HHMMSS>_<voice>.log
logs/<YYYYMMDD_HHMMSS>_text_to_audio_<voice>.log
```

Training logs include full DEBUG output and subprocess logs. Use `--log-file PATH` to write a training run log to a specific file instead of the default timestamped path. Text-to-audio logs include command arguments, selected voice, input/output paths, chunk counts, per-chunk generation timings, output collision handling, final output path, and errors/tracebacks.

---

## Troubleshooting

### torchaudio / torchcodec issues
Avoid installing torchcodec. Use soundfile fallback patch.

### ffmpeg errors
Use standalone static build (conda version causes DLL conflicts).

### CUDA OOM
- Stop inference server before training  
- Reduce LoRA rank if needed  

### Text-to-audio context errors
If llama.cpp reports a context or memory-slot decode error, reduce text generation workload:

```bat
text_to_audio.bat --input <file.txt> --voice hannah --chunk-chars 100 --max-new-tokens 600
```

The converter also retries failing chunks by splitting them smaller, but very long or punctuation-light sections may still require smaller chunk settings.

### Text-to-audio output is not where expected
When `--output` is a directory, files are written there as:

```
<input_stem>_<voice>.wav
```

If the file already exists and overwrite is declined, the converter writes:

```
<input_stem>_<voice>_<HHMMSS>.wav
```

Check the run log for the exact `Saved:` path.

---

## Using with MOSS-TTS

1. Copy `<VoiceName>.yaml` to your server config  
2. Set `REFERENCE_AUDIO` in server config  
3. Restart server  

---

## Project Structure

```
Voice_Gen/
  voice_gen.py
  voice_gen.bat
  text_to_audio.py
  text_to_audio.bat
  train_qlora.py
  logs/
  ffmpeg/
```

---

## Related Work / Future Direction

Voice_Gen is a foundational component of a larger system currently in development:

**Nova-Vex** — a real-time AI character system with personality, memory, and voice-driven interaction.

This pipeline enables consistent voice creation for those characters.

---

## License

(TBD)
