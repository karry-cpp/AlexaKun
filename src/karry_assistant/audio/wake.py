from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

from vosk import KaldiRecognizer, Model

from karry_assistant.audio.mic import MicStream
from karry_assistant.utils.text import (
    contains_wake_phrase,
    remove_wake_phrase,
    strip_leading_filler,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeEvent:
    """Emitted when a wake phrase is detected.

    ``residual_text`` contains anything Vosk transcribed *after* the wake
    phrase in the same utterance. It may be an empty string (user paused
    after saying "hey jimmy") — in that case the caller should switch to
    VAD-based command capture. When non-empty it is only reliable for
    English/ASCII content because Vosk's English model cannot transcribe
    Hindi correctly; the orchestrator can still choose to re-transcribe
    via Whisper for higher fidelity."""

    residual_text: str


class VoskWakeDetector:
    """Loads a Vosk model once and provides a blocking
    :meth:`wait_for_wake` that returns as soon as the wake phrase is seen
    in a final Vosk result."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int,
        wake_phrases: Sequence[str],
        fuzzy_threshold: int = 82,
    ) -> None:
        logger.info("Loading Vosk wake-word model from %s", model_path)
        self._model = Model(model_path)
        self._sample_rate = sample_rate
        self._wake_phrases = tuple(wake_phrases)
        self._threshold = fuzzy_threshold

    def wait_for_wake(
        self,
        mic: MicStream,
        stop_event: threading.Event,
    ) -> Optional[WakeEvent]:
        """Block until a wake phrase is heard or ``stop_event`` is set.

        Returns a :class:`WakeEvent` on detection, or ``None`` on shutdown.
        """
        recognizer = KaldiRecognizer(self._model, self._sample_rate)
        # Ask Vosk to be conservative with endpointing so short utterances
        # like "hey karry" finalize quickly.
        try:
            recognizer.SetWords(False)
        except Exception:  # noqa: BLE001
            pass

        while not stop_event.is_set():
            frame = mic.read(timeout=0.25)
            if frame is None:
                continue

            if recognizer.AcceptWaveform(frame):
                payload = json.loads(recognizer.Result())
                text = payload.get("text", "").strip()
                if not text:
                    continue
                logger.debug("Vosk final: %r", text)
                if contains_wake_phrase(text, self._wake_phrases, self._threshold):
                    residual = strip_leading_filler(
                        remove_wake_phrase(text, self._wake_phrases)
                    )
                    logger.info("Wake phrase detected. Residual: %r", residual)
                    return WakeEvent(residual_text=residual)
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                if partial and contains_wake_phrase(partial, self._wake_phrases, self._threshold):
                    # Force a finalization to grab any trailing command text.
                    final_payload = json.loads(recognizer.FinalResult())
                    final_text = final_payload.get("text", "").strip() or partial
                    residual = strip_leading_filler(
                        remove_wake_phrase(final_text, self._wake_phrases)
                    )
                    logger.info("Wake phrase detected (partial). Residual: %r", residual)
                    return WakeEvent(residual_text=residual)

        return None
