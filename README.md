# Voice_Gen
## Version
Current Version: v0.1.0

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

### Basic

```bat
voice_gen.bat --voice MyVoice --input D:\Audio\raw --output D:\Audio\output
```

### Common Workflows

```bat
# Full run from scratch
voice_gen.bat --voice <VoiceName> --input D:\Training_Data\Audio\<VoiceName> --output D:\Voices\<VoiceName>

# Resume from fine-tuning
voice_gen.bat --from-stage 8

# Zero-shot only (no fine-tuning)
voice_gen.bat --skip-finetune
```

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

Each run generates a timestamped log:

```
logs/<YYYYMMDD_HHMMSS>_<voice>.log
```

Includes full DEBUG output and subprocess logs.

---

## Troubleshooting

### torchaudio / torchcodec issues
Avoid installing torchcodec. Use soundfile fallback patch.

### ffmpeg errors
Use standalone static build (conda version causes DLL conflicts).

### CUDA OOM
- Stop inference server before training  
- Reduce LoRA rank if needed  

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