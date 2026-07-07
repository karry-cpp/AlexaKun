from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# Silence HF Hub's Windows symlink warning — cache falls back to copies,
# which is fine for our use case. Must be set before huggingface_hub imports.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _prepend_cuda_dll_paths() -> None:
    """Make CUDA runtime DLLs from pip packages visible to ctranslate2.

    ``faster-whisper`` (via ``ctranslate2``) loads cuBLAS + cuDNN with
    the raw Win32 ``LoadLibrary`` API, which reads ``PATH`` — not the
    paths registered via ``os.add_dll_directory``. Prepending the pip
    NVIDIA package bin/lib directories to ``PATH`` at import time lets
    the app use GPU without a system-wide CUDA install.
    """
    if not sys.platform.startswith("win"):
        return
    site_pkgs = Path(sys.prefix) / "Lib" / "site-packages"
    # NVIDIA pip packages install under site-packages/nvidia/<lib>/bin
    # (not under a nvidia_<lib>_cu12 directory, contrary to the
    # ``pip install nvidia-<lib>-cu12`` package *name*).
    candidates: list[Path] = []
    nvidia_root = site_pkgs / "nvidia"
    if nvidia_root.is_dir():
        for lib in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            bin_dir = nvidia_root / lib / "bin"
            if bin_dir.is_dir():
                candidates.append(bin_dir)
    if not candidates:
        return
    existing = os.environ.get("PATH", "")
    to_add = [str(p) for p in candidates if str(p) not in existing]
    if not to_add:
        return
    os.environ["PATH"] = os.pathsep.join(to_add) + os.pathsep + existing
    logger.info("Added CUDA runtime dirs to PATH: %s", to_add)


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str  # ISO code, e.g. "en" or "hi"
    duration: float


# Words + phrases Jimmy commands are most likely to contain. Passing
# this to Whisper as ``initial_prompt`` biases the language model
# toward these tokens, which dramatically reduces mistranscriptions
# like "playback to friends" for "play aaoge jab tum" or "Gary" for
# "Jimmy".
_INITIAL_PROMPT = (
    "Hey Jimmy. Play, pause, stop, next, previous, resume song music. "
    "Volume up, volume down, mute, unmute, set volume to. "
    "Open, launch, start Chrome, Firefox, Edge, Notepad, VS Code, "
    "Explorer, Calculator, Settings, Terminal, WhatsApp, Spotify. "
    "Lock, sleep, shutdown, hibernate, restart the PC. "
    "Search on Google, play on YouTube. "
    "Aaoge jab tum, tum hi ho, kesariya, chaiyya chaiyya, kal ho na ho, "
    "chrome kholo, gaana chala do, awaaz kam karo, "
    "PC band kar do, hibernate kar do, restart karo."
)


def _watch_download_size(download_root: str, stop: threading.Event) -> None:
    """Background thread: print a heartbeat with the current on-disk size
    of the Whisper model directory so the user can see download progress."""
    last_reported_mb = -10.0
    root = Path(download_root)
    while not stop.wait(3.0):
        if not root.exists():
            continue
        total = 0
        try:
            for p in root.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except OSError:
            continue
        mb = total / (1024 * 1024)
        # Only print when it changed meaningfully to avoid noise.
        if mb - last_reported_mb >= 5.0:
            print(f"[jimmy] downloading Whisper... {mb:.0f} MB")
            last_reported_mb = mb


class WhisperSTT:
    """Wraps ``faster-whisper`` for offline multilingual transcription.

    The default model is ``large-v3-turbo`` (a distilled version of
    large-v3): near-identical multilingual quality — including Hindi
    and Indian-accented English — but ~4x faster on CPU and ~half the
    disk size. The model is loaded lazily on the first call so startup
    cost only happens when a command actually needs transcribing.
    """

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        compute_type: str = "int8",
        device: str = "auto",
        language: Optional[str] = None,
        download_root: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._device = device
        self._language: Optional[str] = language or None
        self._download_root = download_root
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._device in ("cuda", "auto"):
            _prepend_cuda_dll_paths()
        from faster_whisper import WhisperModel  # local import: heavy dep

        download_root = self._download_root
        if download_root:
            Path(download_root).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading faster-whisper model=%s compute_type=%s device=%s",
            self._model_name,
            self._compute_type,
            self._device,
        )

        # Start a background progress reporter for the (potentially long)
        # first-run download.
        stop_watch = threading.Event()
        watcher: Optional[threading.Thread] = None
        if download_root:
            watcher = threading.Thread(
                target=_watch_download_size,
                args=(download_root, stop_watch),
                daemon=True,
                name="whisper-download-watcher",
            )
            watcher.start()

        started_at = time.monotonic()
        try:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                download_root=download_root,
            )
        finally:
            stop_watch.set()
            if watcher is not None:
                watcher.join(timeout=1.0)

        logger.info("Whisper model loaded in %.1fs", time.monotonic() - started_at)

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> Transcription:
        """Transcribe raw 16-bit mono PCM audio. Returns text + detected
        language."""
        if not pcm_bytes:
            return Transcription(text="", language="en", duration=0.0)

        self._ensure_loaded()

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration = float(audio.size / max(1, sample_rate))
        assert self._model is not None

        segments, info = self._model.transcribe(
            audio,
            language=self._language,   # None → auto-detect (en/hi)
            beam_size=1,
            vad_filter=False,          # we already trimmed via webrtcvad
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=_INITIAL_PROMPT,
            no_speech_threshold=0.45,  # be less eager to declare silence
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        detected = getattr(info, "language", None) or self._language or "en"

        logger.info(
            "Whisper transcript (%s, %.1fs): %r",
            detected,
            duration,
            text,
        )
        return Transcription(text=text, language=detected, duration=duration)

    def warm_up(self) -> None:
        """Run a tiny transcription on synthetic silence so the first
        real command doesn't pay the cold-start cost (cuBLAS JIT, kernel
        compilation, tokenizer load). Safe to call from any thread."""
        try:
            self._ensure_loaded()
            silence = np.zeros(int(0.5 * 16000), dtype=np.int16).tobytes()
            t0 = time.monotonic()
            self.transcribe_pcm(silence, sample_rate=16000)
            logger.info("Whisper warm-up transcribe: %.2fs", time.monotonic() - t0)
        except Exception:  # noqa: BLE001
            logger.exception("Whisper warm-up failed (non-fatal)")
