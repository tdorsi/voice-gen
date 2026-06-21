"""Shared terminal, logging, and prompt helpers for Voice_Gen tools."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _can_encode(text: str) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return True
    except UnicodeEncodeError:
        return False


def console_line(preferred: str, fallback: str = "-", length: int = 60) -> str:
    char = preferred if _can_encode(preferred) else fallback
    return char * length


def console_symbol(preferred: str, fallback: str) -> str:
    return preferred if _can_encode(preferred) else fallback


def setup_logging(
    logger: logging.Logger,
    log_dir: Path,
    run_name: str,
    log_file: Path | None = None,
) -> Path:
    """Configure file + console logging for a Voice_Gen CLI."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{stamp}_{run_name}.log"

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info("Log file: %s", log_file)
    return log_file


def timestamp_for_log() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%H%M%S")


def banner(title: str) -> None:
    line = console_line("═", "=")
    print(f"\n{BOLD}{CYAN}{line}")
    print(f"  {title}")
    print(f"{line}{RESET}\n")


def header(logger: logging.Logger, stage: int, title: str) -> None:
    line = console_line("─", "-")
    logger.info("")
    logger.info(line)
    logger.info("  Stage %d: %s", stage, title)
    logger.info(line)
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  Stage {stage}: {title}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}")


def ok(logger: logging.Logger, msg: str) -> None:
    symbol = console_symbol("✓", "OK")
    logger.info("  %s %s", symbol, msg)
    print(f"{GREEN}  {symbol} {msg}{RESET}")


def warn(logger: logging.Logger, msg: str) -> None:
    logger.warning("  ! %s", msg)
    print(f"{YELLOW}  ! {msg}{RESET}")


def err(logger: logging.Logger, msg: str) -> None:
    symbol = console_symbol("✗", "X")
    logger.error("  %s %s", symbol, msg)
    print(f"{RED}  {symbol} {msg}{RESET}")


def info(logger: logging.Logger, msg: str) -> None:
    logger.info("    %s", msg)


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip().strip('"')
    return value or default
