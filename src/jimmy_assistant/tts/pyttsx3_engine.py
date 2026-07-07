"""Windows SAPI5 TTS via ``pyttsx3``.

Blocking, offline, zero-dep beyond the pyttsx3 wheel. Used as a
fallback when ``edge-tts`` isn't available or fails.

Reliability note
----------------
``pyttsx3.init()`` returns a driver that becomes unusable after a
few ``runAndWait()`` calls when reused across worker threads on
Windows — the second call often silently no-ops. To avoid that, we
create a **fresh engine instance for every** :meth:`speak` call and
dispose of it afterwards. That costs ~150 ms of init time per call
but eliminates the flakiness.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


class Pyttsx3Engine:
    def __init__(self, rate_wpm: int | None = 190) -> None:
        self._rate_wpm = rate_wpm

    def speak(self, text: str, _lang: str = "en") -> bool:
        """Return ``True`` on success, ``False`` if SAPI5 didn't play."""
        text = text.strip()
        if not text:
            return False
        try:
            import pyttsx3  # local import: optional dep

            engine = pyttsx3.init()
            if self._rate_wpm is not None:
                try:
                    engine.setProperty("rate", self._rate_wpm)
                except Exception:  # noqa: BLE001
                    pass
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            return True
        except Exception:  # noqa: BLE001
            logger.exception("pyttsx3 speak failed for %r", text)
            return False
