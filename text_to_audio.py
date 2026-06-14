#!/usr/bin/env python3
"""
Voice_Gen — Text-to-audio conversion with local MOSS-TTS
========================================================

This utility is for inference only. It does not train or fine-tune voices.
Run it from the moss-tts conda environment, or use text_to_audio.bat.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import voice_gen_config
import voice_gen_utils as ui
from voice_gen_utils import BOLD, CYAN, GREEN, RESET


SAMPLE_RATE = 24000
try:
    APP_CONFIG = voice_gen_config.load_config()
except voice_gen_config.ConfigError as exc:
    print(f"Config error: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

MOSS_ROOT = APP_CONFIG.paths.moss_root
MOSS_REPO = APP_CONFIG.paths.moss_repo
LLAMA_CPP_DIR = APP_CONFIG.moss.llama_cpp_dir
LOG_DIR = APP_CONFIG.paths.log_dir
ONNX_DIR = APP_CONFIG.moss.onnx_dir
log = logging.getLogger("text_to_audio")

CONFIGURED_VOICES = APP_CONFIG.voices


def setup_logging(run_name: str = "text_to_audio", log_file: Path | None = None) -> Path:
    return ui.setup_logging(log, LOG_DIR, run_name, log_file)


def banner() -> None:
    ui.banner("Voice_Gen — Text-to-Audio Converter")


def header(stage: int, title: str) -> None:
    ui.header(log, stage, title)


def ok(msg: str) -> None:
    ui.ok(log, msg)


def warn(msg: str) -> None:
    ui.warn(log, msg)


def err(msg: str) -> None:
    ui.err(log, msg)


def info(msg: str) -> None:
    ui.info(log, msg)


def add_windows_dll_paths() -> None:
    """Match the local Windows MOSS server DLL setup."""
    dll_dirs = [
        LLAMA_CPP_DIR,
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"),
        Path(r"C:\Users\thoma\.conda\envs\moss-tts\Lib\site-packages\nvidia\cublas\bin"),
    ]
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
            log.debug("Added DLL directory: %s", dll_dir)

    for dll in ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll"]:
        dll_path = LLAMA_CPP_DIR / dll
        if dll_path.exists():
            ctypes.CDLL(str(dll_path))
            log.debug("Loaded DLL: %s", dll_path)


def check_dependencies() -> None:
    """Verify required packages and runtime files are present before starting."""
    missing: list[str] = []

    for pkg in ("numpy", "soundfile"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"Python package not found: {pkg}  (pip install {pkg})")

    for dll in ("ggml.dll", "llama.dll"):
        if not (LLAMA_CPP_DIR / dll).exists():
            missing.append(f"DLL not found: {LLAMA_CPP_DIR / dll}")

    for onnx_file in ("encoder.onnx", "decoder.onnx"):
        if not (ONNX_DIR / onnx_file).exists():
            missing.append(f"ONNX weight not found: {ONNX_DIR / onnx_file}")

    if missing:
        for msg in missing:
            err(msg)
        raise SystemExit(1)

    ok("Dependencies verified")


def normalize_config_for_windows(config_path: Path) -> Path:
    """Return a config path usable from Windows, normalizing WSL /mnt/d paths."""
    text = config_path.read_text(encoding="utf-8")
    normalized = re.sub(r"/mnt/([a-zA-Z])/", lambda m: f"{m.group(1).upper()}:/", text)
    normalized = normalized.replace("\\", "/")

    if normalized == text:
        return config_path

    runtime_dir = Path(tempfile.gettempdir()) / "voice_gen_text_to_audio"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = runtime_dir / f"{config_path.stem}_windows.yaml"
    normalized_path.write_text(normalized, encoding="utf-8")
    log.info("Normalized config paths for Windows: %s", normalized_path)
    return normalized_path


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def split_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            cut = max(
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
                window.rfind("; "),
                window.rfind(", "),
            )
            if cut < int(max_chars * 0.45):
                cut = window.rfind(" ")
            if cut <= 0:
                cut = max_chars
            else:
                cut += 1
            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].strip()
        if remaining:
            chunks.append(remaining)

    return chunks


def resolve_output_path(input_path: Path, output: Path | None, voice: str) -> Path:
    if output is None:
        return input_path.with_name(f"{input_path.stem}_{voice}.wav")
    if output.suffix.lower() == ".wav":
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{input_path.stem}_{voice}.wav"


def timestamped_output_path(path: Path) -> Path:
    stamp = ui.timestamp_for_filename()
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def synthesize_file(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    reference_path: Path,
    chunk_chars: int,
    max_new_tokens: int,
    silence_ms: int,
    overwrite: bool,
    dry_run: bool,
    show_chunks: bool,
    keep_chunks: bool = False,
) -> tuple[str, Path]:
    header(1, "Preparing text")
    text = read_text(input_path)
    if not text:
        raise ValueError(f"Input file is empty: {input_path}")

    chunks = split_text(text, chunk_chars)
    total_chars = sum(len(chunk) for chunk in chunks)
    info(f"Input      : {input_path}")
    info(f"Output     : {output_path}")
    info(f"Chunks     : {len(chunks)}")
    info(f"Characters : {total_chars}")
    info(f"Chunk chars: {chunk_chars}")
    info(f"Max tokens : {max_new_tokens}")
    info(f"Silence    : {silence_ms} ms")

    if dry_run:
        for idx, chunk in enumerate(chunks, start=1):
            if show_chunks:
                preview = chunk.replace("\n", " ")[:100]
                info(f"{idx:03d}: {len(chunk)} chars | {preview}")
            else:
                info(f"{idx:03d}: {len(chunk)} chars")
        ok("Dry run complete")
        return "dry-run", output_path

    if output_path.exists() and not overwrite:
        original_path = output_path
        output_path = timestamped_output_path(output_path)
        warn(f"Output exists and overwrite was declined: {original_path}")
        info(f"Using timestamped output: {output_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_path}")

    header(2, "Loading voice")
    info(f"Config    : {config_path}")
    info(f"Reference : {reference_path}")

    sys.path.insert(0, str(MOSS_REPO))
    add_windows_dll_paths()

    import numpy as np
    import soundfile as sf
    from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

    config = PipelineConfig.from_yaml(normalize_config_for_windows(config_path))
    silence = np.zeros(int(SAMPLE_RATE * silence_ms / 1000), dtype=np.float32)
    audio_parts: list[np.ndarray] = []

    def generate_chunk(pipeline, chunk: str, label: str) -> list[np.ndarray]:
        try:
            log.debug("%s text length: %d", label, len(chunk))
            waveform = pipeline.generate(
                text=chunk,
                reference_audio=str(reference_path),
                language="en",
                max_new_tokens=max_new_tokens,
            )
            if waveform.size == 0:
                raise RuntimeError(f"{label} produced an empty waveform")
            return [np.asarray(waveform, dtype=np.float32)]
        except RuntimeError as exc:
            msg = str(exc)
            can_split = len(chunk) > 80 and (
                "llama_decode failed" in msg
                or "memory slot" in msg
                or "context" in msg.lower()
            )
            if not can_split:
                log.exception("%s failed and cannot be split safely", label)
                raise
            smaller = split_text(chunk, max(80, len(chunk) // 2))
            if len(smaller) <= 1:
                log.exception("%s failed and split_text did not produce smaller chunks", label)
                raise
            log.warning("%s exceeded context; retrying as %d smaller chunks", label, len(smaller))
            parts: list[np.ndarray] = []
            for sub_idx, sub_chunk in enumerate(smaller, start=1):
                parts.extend(generate_chunk(pipeline, sub_chunk, f"{label}.{sub_idx}"))
                if len(silence):
                    parts.append(silence)
            if parts and len(silence):
                parts.pop()
            return parts

    header(3, "Generating audio")
    start = time.time()
    with LlamaCppPipeline(config) as pipeline:
        for idx, chunk in enumerate(chunks, start=1):
            info(f"Processing chunk {idx} of {len(chunks)} ({len(chunk)} chars)")
            chunk_start = time.time()
            generated_parts = generate_chunk(pipeline, chunk, f"chunk {idx}")
            audio_parts.extend(generated_parts)

            if keep_chunks:
                chunk_path = output_path.with_name(f"{output_path.stem}_chunk_{idx:03d}.wav")
                chunk_audio = np.concatenate(generated_parts) if generated_parts else np.array([], dtype=np.float32)
                sf.write(str(chunk_path), chunk_audio, SAMPLE_RATE)
                log.debug("Saved intermediate chunk: %s", chunk_path)

            if idx < len(chunks) and len(silence):
                audio_parts.append(silence)
            duration = sum(len(part) for part in generated_parts) / SAMPLE_RATE
            elapsed = time.time() - chunk_start
            ok(f"{duration:.2f}s audio in {elapsed:.2f}s")

    header(4, "Writing output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_audio = np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)
    sf.write(str(output_path), final_audio, SAMPLE_RATE)
    ok(f"Saved: {output_path}")
    info(f"Audio duration       : {len(final_audio) / SAMPLE_RATE:.2f}s")
    info(f"Total generation time: {time.time() - start:.2f}s")
    return "saved", output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a text file to WAV audio with local MOSS-TTS voices."
    )
    parser.add_argument("--input", help="Path to a .txt file.")
    parser.add_argument(
        "--output",
        help="Output .wav path, or an output directory when using --voice all.",
    )
    parser.add_argument(
        "--voice",
        default=APP_CONFIG.text_to_audio.default_voice,
        choices=[*CONFIGURED_VOICES.keys(), "all"],
        help="Configured voice preset to use.",
    )
    parser.add_argument("--config", help="Custom MOSS llama.cpp YAML config.")
    parser.add_argument("--reference", help="Custom voice reference WAV.")
    parser.add_argument("--chunk-chars", type=int, default=180)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=900,
        help="Per-chunk generation cap. Lower this if llama.cpp runs out of context.",
    )
    parser.add_argument("--silence-ms", type=int, default=350)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-chunks", action="store_true", help="Print chunk text during --dry-run.")
    parser.add_argument("--keep-chunks", action="store_true", help="Save intermediate numbered WAV files for each chunk.")
    parser.add_argument("--log-file", help="Write detailed logs to this file.")
    return parser.parse_args()


def ask(prompt: str, default: str = "") -> str:
    return ui.ask(prompt, default)


def fill_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input:
        return args

    print("Press Enter to accept defaults.")

    args.input = ask("Input text file", str(APP_CONFIG.paths.default_input_file))

    voice_choices = ", ".join([*CONFIGURED_VOICES.keys(), "all"])
    voice = ask(f"Voice ({voice_choices})", args.voice).lower()
    if voice not in CONFIGURED_VOICES and voice != "all":
        raise ValueError(f"Unknown voice: {voice}")
    args.voice = voice

    output = ask("Output WAV path or directory", args.output or str(APP_CONFIG.paths.default_output_dir))
    args.output = output or args.output

    overwrite = ask("Overwrite existing files? y/N", "N").lower()
    args.overwrite = overwrite in ("y", "yes")
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Check numeric bounds on CLI args before setup_logging is called."""
    errors: list[str] = []
    if args.chunk_chars < 10:
        errors.append(f"--chunk-chars must be >= 10 (got {args.chunk_chars})")
    if args.max_new_tokens < 1:
        errors.append(f"--max-new-tokens must be >= 1 (got {args.max_new_tokens})")
    if args.silence_ms < 0:
        errors.append(f"--silence-ms must be >= 0 (got {args.silence_ms})")
    if errors:
        for msg in errors:
            err(msg)
        raise SystemExit(1)


def main() -> int:
    banner()
    check_dependencies()
    args = fill_interactive_args(parse_args())
    validate_args(args)
    run_name = "text_to_audio"
    if args.voice and args.voice != "all":
        run_name = f"text_to_audio_{args.voice}"
    log_file = setup_logging(
        run_name,
        Path(args.log_file).expanduser().resolve() if args.log_file else None,
    )
    log.info("Log file: %s", log_file)
    log.info("Command: %s", " ".join(sys.argv))
    log.info(ui.console_line("═", "="))
    log.info("Text-to-audio run started")
    log.info("Config: %s", APP_CONFIG.path)
    try:
        voice_gen_config.validate_paths(
            APP_CONFIG,
            ["moss_root", "moss_repo", "voices_dir", "config_dir", "llama_cpp_dir", "onnx_dir"],
            logger=log,
        )
    except voice_gen_config.ConfigError as exc:
        err(str(exc))
        raise SystemExit(1) from exc
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output = Path(args.output).expanduser().resolve() if args.output else None

    if args.config or args.reference:
        if args.voice == "all":
            raise ValueError("--voice all cannot be combined with --config or --reference")
        voices = [args.voice]
        custom_config = Path(args.config).expanduser().resolve() if args.config else None
        custom_reference = Path(args.reference).expanduser().resolve() if args.reference else None
    else:
        voices = list(CONFIGURED_VOICES) if args.voice == "all" else [args.voice]
        custom_config = None
        custom_reference = None

    results: list[tuple[str, Path, str]] = []
    for voice in voices:
        preset = CONFIGURED_VOICES[voice]
        config = custom_config or preset.config
        reference = custom_reference or preset.reference
        output_path = resolve_output_path(input_path, output, voice)
        line = ui.console_line("═", "=")
        log.info("")
        log.info(line)
        log.info("  Voice: %s", voice)
        log.info(line)
        print(f"\n{BOLD}{CYAN}{line}{RESET}")
        print(f"{BOLD}{CYAN}  Voice: {voice}{RESET}")
        print(f"{BOLD}{CYAN}{line}{RESET}")
        status, final_output_path = synthesize_file(
            input_path=input_path,
            output_path=output_path,
            config_path=config,
            reference_path=reference,
            chunk_chars=args.chunk_chars,
            max_new_tokens=args.max_new_tokens,
            silence_ms=args.silence_ms,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            show_chunks=args.show_chunks,
            keep_chunks=args.keep_chunks,
        )
        results.append((voice, final_output_path, status))

    line = ui.console_line("═", "=")
    print(f"\n{BOLD}{GREEN}{line}")
    print("  Text-to-audio conversion complete")
    for voice, path, status in results:
        print(f"  {voice}: {status} -> {path}")
    print(f"  Log   : {log_file}")
    print(f"{line}{RESET}\n")
    log.info("Text-to-audio conversion complete")
    for voice, path, status in results:
        log.info("Result: %s | %s | %s", voice, status, path)
    log.info("Log file: %s", log_file)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        if log.handlers:
            err("Text-to-audio conversion aborted")
            log.error(traceback.format_exc())
        else:
            traceback.print_exc()
        raise SystemExit(1)
