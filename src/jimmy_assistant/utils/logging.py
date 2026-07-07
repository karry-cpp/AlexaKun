from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_CONFIGURED = False


def _app_data_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = str(Path.home() / ".local" / "share")
    path = Path(base) / "Jimmy" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(level: str = "INFO") -> None:
    """Configure a stderr handler and a rotating file handler under
    ``%APPDATA%/Jimmy/logs/jimmy.log``. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        log_file = _app_data_dir() / "jimmy.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Never let logging setup crash the app.
        pass

    # Silence noisy third-party libs.
    for noisy in ("comtypes", "pywhatkit", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def logs_dir() -> Path:
    return _app_data_dir()
