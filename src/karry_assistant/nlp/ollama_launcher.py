"""Try to start the local Ollama server if it's installed but not
already running. Used at Karry startup so the user doesn't have to
manually launch ``ollama serve`` before every session.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008


def _default_exe_locations() -> list[Path]:
    if not sys.platform.startswith("win"):
        return []
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    pf = os.environ.get("ProgramFiles")
    if pf:
        candidates.append(Path(pf) / "Ollama" / "ollama.exe")
    pfx86 = os.environ.get("ProgramFiles(x86)")
    if pfx86:
        candidates.append(Path(pfx86) / "Ollama" / "ollama.exe")
    return candidates


def find_ollama_exe() -> Optional[Path]:
    """Return the path to ``ollama.exe`` if installed anywhere obvious,
    else ``None``."""
    for p in _default_exe_locations():
        if p.is_file():
            return p
    # Fall back to PATH search.
    from shutil import which

    hit = which("ollama")
    return Path(hit) if hit else None


def try_start_ollama(wait_seconds: float = 6.0) -> Optional[int]:
    """If ``ollama.exe`` can be found, launch it as a detached
    ``ollama serve`` process. Return the child PID on success, or
    ``None`` if the exe isn't installed / we couldn't launch it.

    Does NOT verify that the API is reachable — the caller should
    retry :func:`OllamaClient.is_reachable` after a short delay.
    """
    exe = find_ollama_exe()
    if exe is None:
        logger.info("ollama.exe not found in usual locations; skipping auto-start")
        return None

    try:
        creationflags = CREATE_NO_WINDOW | DETACHED_PROCESS if sys.platform.startswith("win") else 0
        proc = subprocess.Popen(
            [str(exe), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
        logger.info("Launched Ollama serve (pid=%s)", proc.pid)
        # Small delay so the API port is bound before the caller probes it.
        time.sleep(min(2.0, max(0.5, wait_seconds / 3)))
        return proc.pid
    except Exception:  # noqa: BLE001
        logger.exception("Failed to launch %s serve", exe)
        return None
