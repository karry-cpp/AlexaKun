"""Edge TTS backend for high-quality Hindi + Indian-English voices.

Synchronously generates an MP3 to a temp file and plays it back via
Windows' MCI (Media Control Interface) which supports MP3 natively —
no extra playback deps required.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import tempfile
import uuid
from pathlib import Path


logger = logging.getLogger(__name__)


class EdgeTtsEngine:
    def __init__(
        self,
        voice_en: str = "en-IN-NeerjaNeural",
        voice_hi: str = "hi-IN-SwaraNeural",
        rate: str = "+0%",
    ) -> None:
        self._voice_en = voice_en
        self._voice_hi = voice_hi
        self._rate = rate

    def _pick_voice(self, lang: str) -> str:
        return self._voice_hi if lang == "hi" else self._voice_en

    async def _synthesize_async(self, text: str, voice: str, out_path: Path) -> None:
        import edge_tts  # local import: optional dep

        communicator = edge_tts.Communicate(text=text, voice=voice, rate=self._rate)
        await communicator.save(str(out_path))

    def speak(self, text: str, lang: str = "en") -> bool:
        """Speak ``text`` via edge-tts. Returns ``True`` on success,
        ``False`` if synthesis or playback failed (caller can fall back)."""
        text = text.strip()
        if not text:
            return False
        voice = self._pick_voice(lang)
        tmpdir = Path(tempfile.gettempdir()) / "jimmy_tts"
        tmpdir.mkdir(parents=True, exist_ok=True)
        # Random ASCII-only filename avoids MCI quoting problems.
        out_path = tmpdir / f"tts_{uuid.uuid4().hex}.mp3"

        try:
            asyncio.run(self._synthesize_async(text, voice, out_path))
        except Exception:  # noqa: BLE001
            logger.exception("edge-tts synthesis failed for %r", text)
            return False

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning("edge-tts produced empty file")
            return False

        ok = _mci_play(out_path)
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return ok


def _mci_play(path: Path) -> bool:
    """Play an MP3 synchronously using Windows MCI. Returns True on success."""
    winmm = ctypes.windll.winmm
    alias = f"jimmy{uuid.uuid4().hex[:8]}"
    open_cmd = f'open "{path}" type mpegvideo alias {alias}'
    play_cmd = f"play {alias} wait"
    close_cmd = f"close {alias}"

    def _send(cmd: str) -> int:
        return int(winmm.mciSendStringW(cmd, None, 0, None))

    rc = _send(open_cmd)
    if rc != 0:
        logger.warning("MCI open failed (rc=%d) for %s", rc, path)
        return False
    try:
        rc = _send(play_cmd)
        if rc != 0:
            logger.warning("MCI play failed (rc=%d)", rc)
            return False
        return True
    finally:
        _send(close_cmd)
