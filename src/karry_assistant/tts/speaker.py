"""Public TTS facade.

Speaks Karry's replies using **edge-tts** as the primary engine — its
Indian-English (``en-IN-NeerjaNeural``) and Hindi (``hi-IN-SwaraNeural``)
neural voices sound natural and consistent for both languages. Falls
back to offline **pyttsx3** (Windows SAPI5) if edge-tts fails or isn't
installed.

Threading model
---------------
Karry's orchestrator runs on a worker thread. TTS synthesis + playback
takes ~1–3 seconds, so a fire-and-forget worker with a bounded queue
avoids blocking the main flow. Use :meth:`speak` for background
announcements; use :meth:`speak_and_wait` when the next step depends
on the phrase being fully heard (confirmation prompts, "Yes?" before
opening the mic for a command).

Failure isolation: any exception in any backend is logged and swallowed.
TTS is a nicety; it must never crash Karry.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional


logger = logging.getLogger(__name__)


class Speaker:
    def __init__(
        self,
        enabled: bool = True,
        voice_en: str = "en-IN-NeerjaNeural",
        voice_hi: str = "hi-IN-SwaraNeural",
        rate: str = "+0%",
    ) -> None:
        self._enabled = enabled
        self._voice_en = voice_en
        self._voice_hi = voice_hi
        self._rate = rate

        self._edge = None
        self._pyttsx3 = None

        # Worker queue. Each item is ``(text, lang, done_event | None)``.
        # A ``None`` sentinel item stops the worker on shutdown.
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run_worker,
            name="karry-tts",
            daemon=True,
        )
        self._worker.start()

    # -- engine lazy-init ------------------------------------------------
    def _get_edge(self):
        if self._edge is None:
            try:
                from karry_assistant.tts.edge_tts_engine import EdgeTtsEngine

                self._edge = EdgeTtsEngine(
                    voice_en=self._voice_en,
                    voice_hi=self._voice_hi,
                    rate=self._rate,
                )
            except Exception:  # noqa: BLE001
                logger.exception("edge-tts unavailable")
                self._edge = False  # sentinel: don't retry
        return self._edge or None

    def _get_pyttsx3(self):
        if self._pyttsx3 is None:
            try:
                from karry_assistant.tts.pyttsx3_engine import Pyttsx3Engine

                self._pyttsx3 = Pyttsx3Engine()
            except Exception:  # noqa: BLE001
                logger.exception("pyttsx3 unavailable")
                self._pyttsx3 = False
        return self._pyttsx3 or None

    # -- public API -----------------------------------------------------
    def speak(self, text: str, lang: Optional[str] = None) -> None:
        """Queue ``text`` for TTS. Returns immediately."""
        if not self._enabled or not text or not text.strip():
            return
        self._queue.put((text, lang or "en", None))

    def speak_and_wait(self, text: str, lang: Optional[str] = None, timeout: float = 15.0) -> None:
        """Queue ``text`` and block until it has finished playing (or
        the timeout elapses)."""
        if not self._enabled or not text or not text.strip():
            return
        done = threading.Event()
        self._queue.put((text, lang or "en", done))
        done.wait(timeout=timeout)

    def stop(self) -> None:
        """Signal the worker to exit and wait briefly for it."""
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    # -- worker ---------------------------------------------------------
    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                logger.debug("TTS worker exiting")
                return
            text, lang, done = item
            try:
                self._speak_one(text, lang)
            except Exception:  # noqa: BLE001
                logger.exception("TTS worker crash for %r", text)
            finally:
                if done is not None:
                    done.set()

    def _speak_one(self, text: str, lang: str) -> None:
        """Try edge-tts first (natural voices), fall back to pyttsx3."""
        edge = self._get_edge()
        if edge is not None:
            try:
                if edge.speak(text, lang=lang):
                    return
            except Exception:  # noqa: BLE001
                logger.exception("edge-tts raised for %r", text)

        pyttsx = self._get_pyttsx3()
        if pyttsx is not None:
            try:
                pyttsx.speak(text, _lang=lang)
            except Exception:  # noqa: BLE001
                logger.exception("pyttsx3 raised for %r", text)
