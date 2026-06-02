"""Configuration loading and validation for Voice_Gen tools."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CONFIG_PATH = Path(__file__).parent / "voice_gen.toml"


class ConfigError(RuntimeError):
    """Raised when Voice_Gen configuration cannot be loaded or validated."""


@dataclass(frozen=True)
class PathConfig:
    moss_root: Path
    moss_repo: Path
    weights_dir: Path
    log_dir: Path
    voices_dir: Path
    default_output_dir: Path
    default_input_file: Path
    ffmpeg_dir: Path


@dataclass(frozen=True)
class MossConfig:
    config_dir: Path
    llama_cpp_dir: Path
    onnx_dir: Path


@dataclass(frozen=True)
class VoiceConfig:
    config: Path
    reference: Path


@dataclass(frozen=True)
class VoiceGenConfig:
    path: Path
    paths: PathConfig
    moss: MossConfig
    voices: dict[str, VoiceConfig]


DEFAULTS = {
    "paths": {
        "moss_root": "D:/AI_Models/Voice/moss-tts",
        "moss_repo": "D:/AI_Models/Voice/moss-tts/repo",
        "weights_dir": "D:/AI_Models/Voice/moss-tts/weights",
        "log_dir": "D:/Development/Voice_Gen/logs",
        "voices_dir": "D:/AI_Models/Voice/moss-tts/voices",
        "default_output_dir": "D:/Training_Data/Audio/TestOut",
        "default_input_file": "D:/Training_Data/Audio/Test_Script/TTS_Script_01.txt",
        "ffmpeg_dir": "D:/Development/Voice_Gen/ffmpeg",
    },
    "moss": {
        "config_dir": "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp",
        "llama_cpp_dir": "D:/AI_Models/Voice/moss-tts/repo/moss_tts_delay/llama_cpp",
        "onnx_dir": "D:/AI_Models/Voice/moss-tts/weights/MOSS-Audio-Tokenizer-ONNX",
    },
    "voices": {
        "lori": {
            "config": "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp/lori.yaml",
            "reference": "D:/AI_Models/Voice/moss-tts/voices/Lori_ref.wav",
        },
        "lilybelle": {
            "config": "D:/AI_Models/Voice/moss-tts/voices/lilybelle.yaml",
            "reference": "D:/AI_Models/Voice/moss-tts/voices/lilybelle_ref_10s.wav",
        },
        "hannah": {
            "config": "D:/AI_Models/Voice/moss-tts/repo/configs/llama_cpp/hannah.yaml",
            "reference": "D:/AI_Models/Voice/moss-tts/voices/Hannah_ref.wav",
        },
    },
}


def load_config(config_path: Path | None = None) -> VoiceGenConfig:
    """Load Voice_Gen configuration from TOML and merge built-in defaults."""
    path = (config_path or CONFIG_PATH).expanduser().resolve()
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}. "
            "Create voice_gen.toml in the Voice_Gen repository root; see README.md."
        )

    try:
        loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    data = _deep_merge(DEFAULTS, loaded)
    return VoiceGenConfig(
        path=path,
        paths=PathConfig(
            moss_root=_path(data, "paths", "moss_root"),
            moss_repo=_path(data, "paths", "moss_repo"),
            weights_dir=_path(data, "paths", "weights_dir"),
            log_dir=_path(data, "paths", "log_dir"),
            voices_dir=_path(data, "paths", "voices_dir"),
            default_output_dir=_path(data, "paths", "default_output_dir"),
            default_input_file=_path(data, "paths", "default_input_file"),
            ffmpeg_dir=_path(data, "paths", "ffmpeg_dir"),
        ),
        moss=MossConfig(
            config_dir=_path(data, "moss", "config_dir"),
            llama_cpp_dir=_path(data, "moss", "llama_cpp_dir"),
            onnx_dir=_path(data, "moss", "onnx_dir"),
        ),
        voices=_voices(data),
    )


def validate_paths(config: VoiceGenConfig, keys: Iterable[str], *, logger=None) -> None:
    """Validate configured paths exist before runtime-heavy operations begin."""
    missing: list[str] = []
    lookup = {
        "moss_root": config.paths.moss_root,
        "moss_repo": config.paths.moss_repo,
        "weights_dir": config.paths.weights_dir,
        "voices_dir": config.paths.voices_dir,
        "default_input_file": config.paths.default_input_file,
        "ffmpeg_dir": config.paths.ffmpeg_dir,
        "config_dir": config.moss.config_dir,
        "llama_cpp_dir": config.moss.llama_cpp_dir,
        "onnx_dir": config.moss.onnx_dir,
    }
    for key in keys:
        try:
            path = lookup[key]
        except KeyError as exc:
            raise ConfigError(f"Unknown config path key requested for validation: {key}") from exc
        if not path.exists():
            missing.append(f"{key}: {path}")

    if missing:
        message = "Configured path(s) not found:\n" + "\n".join(f"  - {item}" for item in missing)
        if logger is not None:
            for item in missing:
                logger.error("Configured path not found: %s", item)
        raise ConfigError(message)


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    merged = {
        key: _deep_merge(value, {}) if isinstance(value, dict) else value
        for key, value in defaults.items()
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _path(data: dict, section: str, key: str) -> Path:
    value = data.get(section, {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid config value: [{section}] {key}")
    return Path(os.path.expandvars(value)).expanduser()


def _voices(data: dict) -> dict[str, VoiceConfig]:
    section = data.get("voices", {})
    if not isinstance(section, dict) or not section:
        raise ConfigError("Missing or invalid config section: [voices]")

    voices: dict[str, VoiceConfig] = {}
    for name, values in section.items():
        if not isinstance(values, dict):
            raise ConfigError(f"Missing or invalid config section: [voices.{name}]")
        config_path = values.get("config")
        reference_path = values.get("reference")
        if not isinstance(config_path, str) or not config_path.strip():
            raise ConfigError(f"Missing or invalid config value: [voices.{name}] config")
        if not isinstance(reference_path, str) or not reference_path.strip():
            raise ConfigError(f"Missing or invalid config value: [voices.{name}] reference")
        voices[name.lower()] = VoiceConfig(
            config=Path(os.path.expandvars(config_path)).expanduser(),
            reference=Path(os.path.expandvars(reference_path)).expanduser(),
        )
    return voices
