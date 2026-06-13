#!/usr/bin/env python3
"""
Voice_Gen — MOSS-TTS voice cloning & fine-tuning pipeline
==========================================================
Stages
  1  Scan input directory, classify files by duration
  2  Split long files (>15 s) into 10-15 s clips at silence boundaries
  3  Noise-reduce and normalise all clips to 24 kHz mono WAV
  4  Score clips by quality; elect best clip as reference WAV
  5  Transcribe with faster-whisper → training JSONL
  6  Verify / download HuggingFace model weights
  7  Encode audio tokens  (prepare_data.py)
  8  Fine-tune            (accelerate launch sft.py, single GPU)
  9  Generate 5 sample outputs from the checkpoint
 10  Write voice YAML config ready for the TTS server

Usage
  python voice_gen.py
  python voice_gen.py --voice MyVoice --input D:/Audio/raw --output D:/Audio/out
  python voice_gen.py --from-stage 5   # resume from a specific stage
  python voice_gen.py --force          # intentionally reuse an existing output dir

Logs are written to D:\\Development\\Voice_Gen\\logs\\<timestamp>.log
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import voice_gen_config
import voice_gen_utils as ui
from voice_gen_utils import BOLD, CYAN, GREEN, RESET

# ── Paths ──────────────────────────────────────────────────────────────────────

try:
    APP_CONFIG = voice_gen_config.load_config()
except voice_gen_config.ConfigError as exc:
    print(f"Config error: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

MOSS_REPO     = APP_CONFIG.paths.moss_repo
WEIGHTS_DIR   = APP_CONFIG.paths.weights_dir
HF_MOSS_DIR   = WEIGHTS_DIR / "MOSS-TTS-HF"
HF_CODEC_DIR  = WEIGHTS_DIR / "MOSS-Audio-Tokenizer-HF"
HF_MOSS_ID    = "OpenMOSS-Team/MOSS-TTS"
HF_CODEC_ID   = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
VOICES_DIR    = APP_CONFIG.paths.voices_dir
SERVER_GGUF   = WEIGHTS_DIR / "MOSS-TTS-GGUF" / "MOSS_TTS_Q4_K_M.gguf"
ONNX_ENC      = APP_CONFIG.moss.onnx_dir / "encoder.onnx"
ONNX_DEC      = APP_CONFIG.moss.onnx_dir / "decoder.onnx"
LOG_DIR       = APP_CONFIG.paths.log_dir
FFMPEG_DIR    = APP_CONFIG.paths.ffmpeg_dir  # standalone static build

# ── Audio constants ────────────────────────────────────────────────────────────

SAMPLE_RATE     = 24000
MIN_CLIP_SECS   = 8
MAX_CLIP_SECS   = 15
TARGET_CLIP     = 12
SILENCE_THRESH  = -35
SILENCE_DUR     = 0.25

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}

SAMPLE_TEXTS = [
    "Hello, this is a test of the voice cloning system.",
    "The quick brown fox jumped over the lazy dog.",
    "I'm really excited to hear how this voice sounds after fine-tuning.",
    "This system allows you to clone any voice from just a few seconds of audio.",
    "Voice generation technology has come a long way in recent years.",
]

CRITICAL_OUTPUT_FILES = (
    "reference.wav",
    "train_raw.jsonl",
    "train_with_codes.jsonl",
    ".voice_gen_state.json",
)
CRITICAL_OUTPUT_DIRS = (
    "clips",
    "checkpoint",
    "samples",
)

# ── Logging setup ──────────────────────────────────────────────────────────────

log = logging.getLogger("voice_gen")

def setup_logging(voice_name: str) -> Path:
    """Configure file + console logging. Returns the log file path."""
    return ui.setup_logging(log, LOG_DIR, voice_name)

def header(stage: int, title: str):
    ui.header(log, stage, title)

def ok(msg: str):
    ui.ok(log, msg)

def warn(msg: str):
    ui.warn(log, msg)

def err(msg: str):
    ui.err(log, msg)

def info(msg: str):
    ui.info(log, msg)

# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

NULL = "NUL"

def _run_ffmpeg(cmd: list, capture: bool = True) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command. On failure, log full stderr and re-raise."""
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            capture_output=capture,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg command failed: %s", " ".join(str(c) for c in cmd))
        if e.stdout:
            log.error("  stdout: %s", e.stdout.strip())
        if e.stderr:
            log.error("  stderr: %s", e.stderr.strip())
        raise

def duration(path: Path) -> float:
    """Return audio duration in seconds."""
    log.debug("duration check: %s", path)
    r = _run_ffmpeg([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    raw = r.stdout.strip()
    if not raw:
        raise ValueError(f"ffprobe returned empty duration for {path}")
    return float(raw)

def rms_db(path: Path) -> float:
    """Return mean_volume dBFS."""
    log.debug("rms_db: %s", path)
    r = subprocess.run(
        [FFMPEG_BIN, "-i", str(path), "-af", "volumedetect",
         "-vn", "-sn", "-dn", "-f", "null", NULL],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", r.stderr)
    val = float(m.group(1)) if m else -99.0
    log.debug("  rms_db = %.1f dB", val)
    return val

def silence_ratio(path: Path) -> float:
    """Fraction of the file that is silent."""
    log.debug("silence_ratio: %s", path)
    r = subprocess.run(
        [FFMPEG_BIN, "-i", str(path),
         "-af", f"silencedetect=noise={SILENCE_THRESH}dB:d=0.1",
         "-f", "null", NULL],
        capture_output=True, text=True,
    )
    starts = re.findall(r"silence_start: ([\d.]+)", r.stderr)
    ends   = re.findall(r"silence_end: ([\d.]+)", r.stderr)
    silent_secs = sum(float(e) - float(s) for s, e in zip(starts, ends))
    dur = duration(path)
    ratio = (silent_secs / dur) if dur > 0 else 1.0
    log.debug("  silence_ratio = %.2f (%.2fs silent of %.2fs)", ratio, silent_secs, dur)
    return ratio

def detect_silences(path: Path) -> list:
    """Return list of (start, end) silence intervals."""
    log.debug("detect_silences: %s", path)
    r = subprocess.run(
        [FFMPEG_BIN, "-i", str(path),
         "-af", f"silencedetect=noise={SILENCE_THRESH}dB:d={SILENCE_DUR}",
         "-f", "null", NULL],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    pairs  = list(zip(starts, ends[:len(starts)]))
    log.debug("  found %d silence intervals", len(pairs))
    return pairs

def ffmpeg_extract(src: Path, dst: Path, start: float, dur: float):
    """Extract a segment from src, normalise, write 24 kHz mono WAV."""
    log.debug("extract: %s [%.2f + %.2f] → %s", src.name, start, dur, dst.name)
    _run_ffmpeg([
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ])

def ffmpeg_clean(src: Path, dst: Path):
    """Noise-reduce (afftdn) + loudnorm + resample → 24 kHz mono WAV."""
    log.debug("clean: %s → %s", src.name, dst.name)
    _run_ffmpeg([
        FFMPEG_BIN, "-y", "-i", str(src),
        "-af", "afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ])

def ffmpeg_to_wav(src: Path, dst: Path):
    """Convert any audio → 24 kHz mono WAV (normalise only, no denoising)."""
    log.debug("to_wav: %s → %s", src.name, dst.name)
    _run_ffmpeg([
        FFMPEG_BIN, "-y", "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ])

# ── Split algorithm ────────────────────────────────────────────────────────────

def split_file(src: Path, clips_dir: Path, stem: str) -> list[Path]:
    """Split src into MIN_CLIP_SECS–MAX_CLIP_SECS clips at silence boundaries."""
    total    = duration(src)
    silences = detect_silences(src)
    log.info("    Splitting %s (%.1fs, %d silence intervals)", src.name, total, len(silences))

    silence_mids = [(s + e) / 2 for s, e in silences]
    clips, cursor, clip_idx = [], 0.0, 0

    while cursor < total - MIN_CLIP_SECS:
        target_end = cursor + TARGET_CLIP
        max_end    = cursor + MAX_CLIP_SECS
        min_end    = cursor + MIN_CLIP_SECS

        candidates = [m for m in silence_mids if min_end <= m <= max_end]
        if candidates:
            cut = min(candidates, key=lambda m: abs(m - target_end))
            log.debug("    clip %03d: silence cut at %.2f", clip_idx, cut)
        else:
            cut = min(target_end, total)
            log.debug("    clip %03d: no silence found, cutting at %.2f", clip_idx, cut)

        clip_dur = cut - cursor
        if clip_dur < MIN_CLIP_SECS:
            log.debug("    clip %03d: remaining %.2fs < min %.2fs, stopping", clip_idx, clip_dur, MIN_CLIP_SECS)
            break

        out = clips_dir / f"{stem}_clip{clip_idx:03d}.wav"
        try:
            ffmpeg_extract(src, out, cursor, clip_dur)
            clips.append(out)
            info(f"  clip {clip_idx:03d}: {cursor:.1f}s → {cut:.1f}s ({clip_dur:.1f}s)")
        except Exception:
            log.error("    Failed to extract clip %03d from %s", clip_idx, src.name)
            log.debug(traceback.format_exc())

        cursor = cut
        clip_idx += 1

    return clips

# ── Quality scoring ────────────────────────────────────────────────────────────

def score_clip(path: Path) -> float:
    """Higher = better quality reference candidate."""
    rms    = rms_db(path)
    sr     = silence_ratio(path)
    dur    = duration(path)
    dur_pen = max(0.0, (TARGET_CLIP - dur) / TARGET_CLIP) * 10
    score  = rms - (sr * 20) - dur_pen
    log.debug("score_clip %s: rms=%.1f sr=%.2f dur=%.1f pen=%.1f → %.1f",
              path.name, rms, sr, dur, dur_pen, score)
    return score

# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe_clips(clips: list[Path]) -> dict[Path, str]:
    """Return {clip_path: transcript_text} using faster-whisper."""
    from faster_whisper import WhisperModel
    info("Loading Whisper model (base.en)…")
    try:
        model = WhisperModel("base.en", device="cuda", compute_type="float16")
    except Exception:
        warn("CUDA Whisper failed, falling back to CPU…")
        log.debug(traceback.format_exc())
        model = WhisperModel("base.en", device="cpu", compute_type="int8")

    results = {}
    for clip in clips:
        try:
            segments, info_meta = model.transcribe(str(clip), language="en")
            text = " ".join(s.text.strip() for s in segments).strip()
            results[clip] = text
            log.info("    %s → %s", clip.name, text[:100])
        except Exception:
            log.error("    Transcription failed for %s", clip.name)
            log.debug(traceback.format_exc())
            results[clip] = ""
    return results

# ── Weight download ────────────────────────────────────────────────────────────

def ensure_weights():
    from huggingface_hub import snapshot_download

    if not HF_MOSS_DIR.exists() or not any(HF_MOSS_DIR.glob("*.safetensors")):
        warn("Full MOSS-TTS weights not found — downloading…")
        log.info("    Downloading %s → %s", HF_MOSS_ID, HF_MOSS_DIR)
        snapshot_download(
            repo_id=HF_MOSS_ID,
            local_dir=str(HF_MOSS_DIR),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        ok(f"MOSS-TTS weights → {HF_MOSS_DIR}")
    else:
        ok(f"MOSS-TTS weights present: {HF_MOSS_DIR}")

    if not HF_CODEC_DIR.exists() or not any(HF_CODEC_DIR.iterdir()):
        warn("MOSS-Audio-Tokenizer weights not found — downloading…")
        log.info("    Downloading %s → %s", HF_CODEC_ID, HF_CODEC_DIR)
        snapshot_download(repo_id=HF_CODEC_ID, local_dir=str(HF_CODEC_DIR))
        ok(f"Codec weights → {HF_CODEC_DIR}")
    else:
        ok(f"Codec weights present: {HF_CODEC_DIR}")

# ── Subprocess runner ──────────────────────────────────────────────────────────

def run_cmd(cmd: list, cwd: Path = None, label: str = ""):
    """
    Run a command, streaming stdout/stderr live to the console and log file.
    On failure, the captured output is written to the log before re-raising.
    """
    if label:
        info(label)
    cmd_str = " ".join(str(c) for c in cmd)
    log.info("    $ %s", cmd_str)
    print(f"  $ {cmd_str}\n")

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        print(line)
        log.debug("    %s", line)

    proc.wait()
    if proc.returncode != 0:
        log.error("Command failed (exit %d): %s", proc.returncode, cmd_str)
        log.error("--- subprocess output ---")
        for line in output_lines:
            log.error("  %s", line)
        log.error("--- end subprocess output ---")
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# ── Stage functions ────────────────────────────────────────────────────────────

def stage1_scan(input_dir: Path) -> tuple[list[Path], list[Path]]:
    header(1, "Scanning input directory")
    log.info("    Input dir: %s", input_dir)

    all_files = [
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    ]
    log.info("    Found %d audio file(s) with extensions %s", len(all_files), AUDIO_EXTS)

    if not all_files:
        err(f"No audio files found in {input_dir}")
        log.error("Supported extensions: %s", AUDIO_EXTS)
        sys.exit(1)

    short, long_ = [], []
    for f in all_files:
        log.debug("  Checking: %s", f)
        try:
            d = duration(f)
        except Exception:
            log.error("  Could not read duration for: %s", f)
            log.debug(traceback.format_exc())
            warn(f"  {f.name}: could not read — skipping (see log for details)")
            continue

        log.info("    %s: %.2fs", f.name, d)
        if d < MIN_CLIP_SECS:
            warn(f"  {f.name}: {d:.1f}s — too short (min {MIN_CLIP_SECS}s), skipping")
        elif d > MAX_CLIP_SECS:
            info(f"  {f.name}: {d:.1f}s — needs splitting")
            long_.append(f)
        else:
            info(f"  {f.name}: {d:.1f}s — OK")
            short.append(f)

    ok(f"{len(short)} short clips, {len(long_)} files to split")
    if not short and not long_:
        err("No usable audio files found — all were too short or unreadable")
        sys.exit(1)
    return short, long_


def stage2_split(long_files: list[Path], clips_dir: Path) -> list[Path]:
    header(2, "Splitting long files")
    new_clips = []
    for src in long_files:
        info(f"Splitting {src.name}…")
        try:
            clips = split_file(src, clips_dir, src.stem)
            ok(f"  → {len(clips)} clips from {src.name}")
            new_clips.extend(clips)
        except Exception:
            log.error("  Failed to split %s", src.name)
            log.debug(traceback.format_exc())
            err(f"  {src.name}: split failed — see log for details")
    return new_clips


def stage3_clean(short_files: list[Path], split_clips: list[Path],
                 clips_dir: Path) -> list[Path]:
    header(3, "Noise reduction & normalisation")
    all_clips = []

    for src in short_files:
        dst = clips_dir / f"{src.stem}_clean.wav"
        info(f"Cleaning {src.name} → {dst.name}")
        try:
            ffmpeg_clean(src, dst)
            all_clips.append(dst)
        except Exception:
            log.error("  ffmpeg_clean failed for %s", src.name)
            log.debug(traceback.format_exc())
            err(f"  {src.name}: noise reduction failed — see log")

    for src in split_clips:
        dst = clips_dir / f"{src.stem}_n.wav"
        info(f"Normalising {src.name} → {dst.name}")
        try:
            ffmpeg_to_wav(src, dst)
            src.unlink()
            all_clips.append(dst)
        except Exception:
            log.error("  ffmpeg_to_wav failed for %s", src.name)
            log.debug(traceback.format_exc())
            err(f"  {src.name}: normalisation failed — see log")

    if not all_clips:
        err("No clips survived stage 3 — check log for ffmpeg errors")
        sys.exit(1)

    ok(f"{len(all_clips)} clips ready")
    return all_clips


def stage4_select_reference(clips: list[Path], output_dir: Path) -> Path:
    header(4, "Selecting reference clip")
    scores = {}
    for clip in clips:
        try:
            s = score_clip(clip)
            scores[clip] = s
            info(f"  {clip.name}: score={s:.1f}")
        except Exception:
            log.error("  Scoring failed for %s", clip.name)
            log.debug(traceback.format_exc())
            warn(f"  {clip.name}: could not score — skipping")

    if not scores:
        err("No clips could be scored — check log")
        sys.exit(1)

    best = max(scores, key=scores.__getitem__)
    ref  = output_dir / "reference.wav"
    shutil.copy2(best, ref)
    ok(f"Reference: {best.name} (score={scores[best]:.1f}) → {ref}")
    log.info("    Reference selected: %s (score=%.1f)", best.name, scores[best])
    return ref


def stage5_transcribe(clips: list[Path], ref_wav: Path, output_dir: Path) -> Path:
    header(5, "Transcription → JSONL")
    transcripts = transcribe_clips(clips)

    jsonl_path = output_dir / "train_raw.jsonl"
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for clip, text in transcripts.items():
            if not text:
                warn(f"  Empty transcript for {clip.name}, skipping")
                continue
            record = {
                "audio":     str(clip),
                "text":      text,
                "ref_audio": str(ref_wav),
                "language":  "en",
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    if written == 0:
        err("No transcripts written — all clips returned empty text")
        sys.exit(1)

    ok(f"Wrote {written} entries → {jsonl_path}")
    return jsonl_path


def stage6_weights(skip: bool = False):
    header(6, "HuggingFace weights")
    if skip:
        info("Skipping download check (--skip-download)")
        return
    ensure_weights()


def stage7_prepare_data(jsonl_raw: Path, output_dir: Path) -> Path:
    header(7, "Encoding audio tokens (prepare_data.py)")
    jsonl_out = output_dir / "train_with_codes.jsonl"
    run_cmd(
        [
            sys.executable,
            str(MOSS_REPO / "moss_tts_delay" / "finetuning" / "prepare_data.py"),
            "--model-path", str(HF_MOSS_DIR),
            "--codec-path", str(HF_CODEC_DIR),
            "--device", "auto",
            "--input-jsonl", str(jsonl_raw),
            "--output-jsonl", str(jsonl_out),
        ],
        cwd=MOSS_REPO,
        label="Encoding audio → token codes…",
    )
    ok(f"Prepared JSONL → {jsonl_out}")
    return jsonl_out


def stage8_finetune(jsonl_codes: Path, checkpoint_dir: Path):
    header(8, "Fine-tuning (QLoRA 4-bit, single GPU)")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).parent / "train_qlora.py"
    run_cmd(
        [
            sys.executable, str(train_script),
            "--model-path",                str(HF_MOSS_DIR),
            "--codec-path",                str(HF_CODEC_DIR),
            "--train-jsonl",               str(jsonl_codes),
            "--output-dir",                str(checkpoint_dir),
            "--epochs",                    "3",
            "--lr",                        "1e-4",
            "--lora-r",                    "16",
            "--lora-alpha",                "32",
            "--grad-accum",                "8",
            "--channelwise-loss-weight",   "1,32",
        ],
        cwd=MOSS_REPO,
        label="Launching QLoRA fine-tuning (4-bit, fits 12 GB VRAM)…",
    )
    ok(f"Fine-tuning complete → {checkpoint_dir}")


def stage9_samples(checkpoint_dir: Path, ref_wav: Path, samples_dir: Path):
    header(9, "Generating sample outputs")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # QLoRA saves a LoRA adapter; check adapter_path.txt written by train_qlora.py
    adapter_path_file = checkpoint_dir / "adapter_path.txt"
    if adapter_path_file.exists():
        model_path = adapter_path_file.read_text().strip()
    else:
        # Fallback: last adapter-* dir, or checkpoint-epoch-* for old sft.py runs
        adapters = sorted(checkpoint_dir.glob("adapter-*"))
        sft_ckpts = sorted(checkpoint_dir.glob("checkpoint-epoch-*"))
        candidates = adapters or sft_ckpts
        model_path = str(candidates[-1]) if candidates else str(checkpoint_dir)
    info(f"Using checkpoint: {model_path}")
    log.info("    Checkpoint: %s", model_path)

    gen_script = Path(__file__).parent / "_gen_sample.py"
    _write_gen_script(gen_script)

    for i, text in enumerate(SAMPLE_TEXTS):
        out_wav = samples_dir / f"sample_{i+1:02d}.wav"
        try:
            run_cmd(
                [
                    sys.executable, str(gen_script),
                    "--base-model-path", str(HF_MOSS_DIR),
                    "--adapter-path",    model_path,
                    "--ref-audio",       str(ref_wav),
                    "--text",            text,
                    "--output",          str(out_wav),
                ],
                cwd=MOSS_REPO,
                label=f"Sample {i+1}: {text[:60]}",
            )
            ok(f"→ {out_wav.name}")
        except Exception:
            log.error("  Sample %d generation failed", i + 1)
            log.debug(traceback.format_exc())
            err(f"  Sample {i+1} failed — see log")

    gen_script.unlink(missing_ok=True)


def stage10_config(voice_name: str, ref_wav: Path, checkpoint_dir: Path,
                   output_dir: Path) -> Path:
    header(10, "Writing voice config")
    import yaml

    config = {
        "backbone_gguf":        str(SERVER_GGUF),
        "audio_backend":        "onnx",
        "audio_encoder_onnx":   str(ONNX_ENC),
        "audio_decoder_onnx":   str(ONNX_DEC),
        "n_ctx":                2048,
        "n_gpu_layers":         -1,
        "kv_cache_type_k":      "q8_0",
        "kv_cache_type_v":      "q8_0",
        "use_gpu_audio":        True,
        "reference_audio":      str(ref_wav),
        "voice_name":           voice_name,
        "finetuned_checkpoint": str(checkpoint_dir),
    }

    cfg_path = output_dir / f"{voice_name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    shared_ref = VOICES_DIR / f"{voice_name}_ref.wav"
    try:
        shutil.copy2(ref_wav, shared_ref)
        ok(f"Reference WAV copied → {shared_ref}")
        log.info("    Shared reference: %s", shared_ref)
    except Exception:
        log.error("  Could not copy reference to voices dir")
        log.debug(traceback.format_exc())
        warn(f"  Could not copy to {shared_ref} — copy manually if needed")

    ok(f"Config → {cfg_path}")
    log.info("    Config written: %s", cfg_path)
    return cfg_path

# ── Inline sample-generation helper script ─────────────────────────────────────

def _write_gen_script(path: Path):
    moss_repo_str = str(MOSS_REPO).replace("\\", "\\\\")
    path.write_text(f'''\
"""Minimal single-sample inference from a fine-tuned MOSS-TTS LoRA checkpoint."""
import argparse, importlib.util, sys
from pathlib import Path

# Ensure moss_tts_delay is importable
_repo = r"{moss_repo_str}"
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import torch, torchaudio
from transformers import AutoProcessor
from peft import PeftModel
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

def resolve_attn(device, dtype):
    if device == "cuda" and importlib.util.find_spec("flash_attn") and dtype in (torch.float16, torch.bfloat16):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa" if device == "cuda" else "eager"

p = argparse.ArgumentParser()
p.add_argument("--base-model-path", required=True)
p.add_argument("--adapter-path",    required=True)
p.add_argument("--ref-audio",       required=True)
p.add_argument("--text",            required=True)
p.add_argument("--output",          required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32
attn   = resolve_attn(device, dtype)

# Load processor from base model (has tokenizer + audio tokenizer config)
processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

# Load base model then apply LoRA adapter
base_model = MossTTSDelayModel.from_pretrained(
    args.base_model_path, torch_dtype=dtype, attn_implementation=attn
)
model = PeftModel.from_pretrained(base_model, args.adapter_path)
model = model.merge_and_unload()  # merge LoRA into weights for clean inference
model = model.to(device).eval()

conversation = [[processor.build_user_message(text=args.text, reference=[args.ref_audio])]]
batch   = processor(conversation, mode="generation")
outputs = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=4096,
)
message = processor.decode(outputs)[0]
audio   = message.audio_codes_list[0]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
torchaudio.save(args.output, audio.unsqueeze(0), processor.model_config.sampling_rate)
print("Saved: " + args.output)
''', encoding="utf-8")

# ── State persistence ──────────────────────────────────────────────────────────

def save_state(state_file: Path, state: dict):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}

# ── Main ───────────────────────────────────────────────────────────────────────

def ask(prompt: str, default: str = "") -> str:
    return ui.ask(prompt, default)

def parse_args():
    p = argparse.ArgumentParser(description="MOSS-TTS voice cloning & fine-tuning pipeline")
    p.add_argument("--voice",         help="Voice name")
    p.add_argument("--input",         help="Directory of raw audio training files")
    p.add_argument("--output",        help="Directory for all generated output")
    p.add_argument("--from-stage",    type=int, default=1, metavar="N",
                   help="Resume from stage N (1-10)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip HF weight download check (stage 6)")
    p.add_argument("--skip-finetune", action="store_true",
                   help="Skip fine-tuning (stages 7-9)")
    p.add_argument("--force",         action="store_true",
                   help="Allow a fresh run to reuse an existing output directory")
    return p.parse_args()


def _find_binary(name: str) -> str:
    """
    Return the full path to ffmpeg or ffprobe.
    Preference order:
      1. Standalone static build in Voice_Gen/ffmpeg/ (no DLL conflicts)
      2. System PATH
    """
    import shutil
    # Search the standalone build directory tree first
    if FFMPEG_DIR.exists():
        matches = list(FFMPEG_DIR.rglob(f"{name}.exe"))
        if matches:
            return str(matches[0])
    # Fall back to PATH
    found = shutil.which(name)
    if found:
        return found
    return name   # will fail at runtime with a clear error

# Resolved binary paths (set once at import time)
FFMPEG_BIN  = _find_binary("ffmpeg")
FFPROBE_BIN = _find_binary("ffprobe")


def check_dependencies():
    """Verify ffmpeg and ffprobe are reachable before starting."""
    log.info("ffmpeg  : %s", FFMPEG_BIN)
    log.info("ffprobe : %s", FFPROBE_BIN)

    for label, binary in (("ffmpeg", FFMPEG_BIN), ("ffprobe", FFPROBE_BIN)):
        try:
            r = subprocess.run([binary, "-version"], capture_output=True, timeout=10)
            if r.returncode != 0:
                raise RuntimeError(f"exit {r.returncode}")
            log.debug("%s version OK", label)
        except Exception as e:
            err(f"{label} not working ({binary}): {e}")
            log.error("Dependency check failed for %s at %s: %s", label, binary, e)
            print(f"  Download standalone build:  https://github.com/BtbN/FFmpeg-Builds/releases")
            print(f"  Extract to: D:\\Development\\Voice_Gen\\ffmpeg\\")
            sys.exit(1)

    ok(f"ffmpeg : {FFMPEG_BIN}")
    ok(f"ffprobe: {FFPROBE_BIN}")


def find_output_collisions(output_dir: Path) -> list[Path]:
    """Return existing output paths that make a fresh run non-destructive unsafe."""
    if not output_dir.exists():
        return []

    collisions: list[Path] = [output_dir]
    for name in CRITICAL_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            collisions.append(path)
    for name in CRITICAL_OUTPUT_DIRS:
        path = output_dir / name
        if path.exists():
            collisions.append(path)
    return collisions


def enforce_output_protection(output_dir: Path, from_stage: int, force: bool):
    """Prevent fresh runs from reusing an existing output directory by default."""
    if from_stage > 1:
        info(f"Resume mode (--from-stage {from_stage}); existing output directory is allowed")
        log.info("Overwrite protection: resume mode allows existing output directory: %s", output_dir)
        return

    collisions = find_output_collisions(output_dir)
    if not collisions:
        return

    if force:
        warn(f"--force set; reusing existing output directory: {output_dir}")
        log.warning("Overwrite protection overridden with --force for output directory: %s", output_dir)
        for path in collisions:
            log.warning("  Existing output path: %s", path)
        return

    err(f"Output directory already exists: {output_dir}")
    log.error("Overwrite protection stopped fresh run. Existing output path(s):")
    for path in collisions:
        log.error("  %s", path)
    print()
    print("Refusing to reuse an existing output directory for a fresh run.")
    print("Use --from-stage N to resume an existing run, or --force to intentionally reuse this output path.")
    sys.exit(1)


def main():
    args = parse_args()

    ui.banner("Voice_Gen — MOSS-TTS Voice Cloning Pipeline")

    check_dependencies()

    voice_name = args.voice or ask("Voice name", "MyVoice")
    input_dir  = Path(args.input  or ask("Input audio directory"))
    output_dir = Path(args.output or ask(
        "Output directory",
        str(APP_CONFIG.paths.default_output_dir / voice_name),
    ))

    # Logging starts here — voice_name is known
    log_file = setup_logging(voice_name)

    log.info(ui.console_line("═", "="))
    log.info("Voice_Gen run started")
    log.info("  Config     : %s", APP_CONFIG.path)
    log.info("  Voice name : %s", voice_name)
    log.info("  Input dir  : %s", input_dir)
    log.info("  Output dir : %s", output_dir)
    log.info("  From stage : %d", args.from_stage)
    log.info("  Force      : %s", args.force)
    log.info(ui.console_line("═", "="))
    try:
        voice_gen_config.validate_paths(
            APP_CONFIG,
            ["moss_repo", "weights_dir", "voices_dir", "onnx_dir"],
            logger=log,
        )
    except voice_gen_config.ConfigError as exc:
        err(str(exc))
        raise SystemExit(1) from exc

    if not input_dir.exists():
        err(f"Input directory not found: {input_dir}")
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    enforce_output_protection(output_dir, args.from_stage, args.force)

    clips_dir      = output_dir / "clips"
    samples_dir    = output_dir / "samples"
    checkpoint_dir = output_dir / "checkpoint"
    clips_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    info(f"Voice name : {voice_name}")
    info(f"Input      : {input_dir}")
    info(f"Output     : {output_dir}")
    info(f"Log file   : {log_file}")

    state_file = output_dir / ".voice_gen_state.json"
    state = load_state(state_file) if args.from_stage > 1 else {}

    try:
        # ── Stage 1 ──
        if args.from_stage <= 1:
            short, long_ = stage1_scan(input_dir)
            state.update({"short": [str(p) for p in short],
                          "long":  [str(p) for p in long_]})
            save_state(state_file, state)
        else:
            short = [Path(p) for p in state.get("short", [])]
            long_ = [Path(p) for p in state.get("long",  [])]
            ok("Stage 1 loaded from saved state")

        # ── Stage 2 ──
        if args.from_stage <= 2:
            split_clips = stage2_split(long_, clips_dir)
            state["split_clips"] = [str(p) for p in split_clips]
            save_state(state_file, state)
        else:
            split_clips = [Path(p) for p in state.get("split_clips", [])]
            ok("Stage 2 loaded from saved state")

        # ── Stage 3 ──
        if args.from_stage <= 3:
            all_clips = stage3_clean(short, split_clips, clips_dir)
            state["all_clips"] = [str(p) for p in all_clips]
            save_state(state_file, state)
        else:
            all_clips = [Path(p) for p in state.get("all_clips", [])]
            ok("Stage 3 loaded from saved state")

        # ── Stage 4 ──
        if args.from_stage <= 4:
            ref_wav = stage4_select_reference(all_clips, output_dir)
            state["ref_wav"] = str(ref_wav)
            save_state(state_file, state)
        else:
            ref_wav = Path(state.get("ref_wav", output_dir / "reference.wav"))
            ok("Stage 4 loaded from saved state")

        # ── Stage 5 ──
        if args.from_stage <= 5:
            jsonl_raw = stage5_transcribe(all_clips, ref_wav, output_dir)
            state["jsonl_raw"] = str(jsonl_raw)
            save_state(state_file, state)
        else:
            jsonl_raw = Path(state.get("jsonl_raw", output_dir / "train_raw.jsonl"))
            ok("Stage 5 loaded from saved state")

        if args.skip_finetune:
            warn("--skip-finetune set; skipping stages 6-9")
        else:
            # ── Stage 6 ──
            if args.from_stage <= 6:
                stage6_weights(skip=args.skip_download)
                save_state(state_file, state)

            # ── Stage 7 ──
            if args.from_stage <= 7:
                jsonl_codes = stage7_prepare_data(jsonl_raw, output_dir)
                state["jsonl_codes"] = str(jsonl_codes)
                save_state(state_file, state)
            else:
                jsonl_codes = Path(state.get("jsonl_codes",
                                             output_dir / "train_with_codes.jsonl"))
                ok("Stage 7 loaded from saved state")

            # ── Stage 8 ──
            if args.from_stage <= 8:
                stage8_finetune(jsonl_codes, checkpoint_dir)
                save_state(state_file, state)

            # ── Stage 9 ──
            if args.from_stage <= 9:
                stage9_samples(checkpoint_dir, ref_wav, samples_dir)
                save_state(state_file, state)

        # ── Stage 10 ──
        if args.from_stage <= 10:
            cfg_path = stage10_config(voice_name, ref_wav, checkpoint_dir, output_dir)
            state["config"] = str(cfg_path)
            save_state(state_file, state)

    except SystemExit:
        raise
    except Exception:
        log.error("Unhandled exception — pipeline aborted")
        log.error(traceback.format_exc())
        err("Pipeline aborted — full traceback written to log:")
        err(f"  {log_file}")
        sys.exit(1)

    final_line = ui.console_line("═", "=")
    print(f"\n{BOLD}{GREEN}{final_line}")
    print(f"  Voice '{voice_name}' pipeline complete!")
    print(f"  Reference : {ref_wav}")
    print(f"  Samples   : {samples_dir}")
    if not args.skip_finetune:
        print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Config    : {output_dir / (voice_name + '.yaml')}")
    print(f"  Log       : {log_file}")
    print(f"{final_line}{RESET}\n")

    log.info("Pipeline complete for voice '%s'", voice_name)


def run_cli():
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)


if __name__ == "__main__":
    run_cli()
