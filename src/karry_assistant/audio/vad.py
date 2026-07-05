from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from karry_assistant.audio.mic import MicStream


logger = logging.getLogger(__name__)


class VadCommandRecorder:
    """Reads frames from a :class:`MicStream` and accumulates PCM audio
    while the user is speaking. Uses ``webrtcvad`` if available; falls
    back to a simple RMS-energy gate so the app still works if webrtcvad
    wheels are not installable for the current Python version.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        silence_ms: int = 900,
        max_seconds: float = 12.0,
        preroll_ms: int = 300,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_frames_target = max(1, silence_ms // frame_ms)
        self._max_frames = int(max_seconds * 1000 / frame_ms)
        self._preroll_bytes = int((preroll_ms / 1000) * sample_rate) * 2

        self._vad = None
        try:
            import webrtcvad  # local import: optional dep

            self._vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))
            logger.info("Using webrtcvad (aggressiveness=%d)", aggressiveness)
        except Exception as exc:  # noqa: BLE001
            logger.warning("webrtcvad unavailable (%s); falling back to RMS gate", exc)

    # -- helpers ---------------------------------------------------------
    def _is_speech(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self._sample_rate)
            except Exception:  # noqa: BLE001
                return self._rms_gate(frame)
        return self._rms_gate(frame)

    @staticmethod
    def _rms_gate(frame: bytes, threshold: float = 500.0) -> bool:
        """Cheap RMS-energy VAD fallback."""
        if not frame:
            return False
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples * samples)))
        return rms >= threshold

    # -- public ---------------------------------------------------------
    def record(
        self,
        mic: MicStream,
        stop_event: threading.Event,
        prepend_audio: Optional[bytes] = None,
    ) -> bytes:
        """Capture speech until ``silence_ms`` of silence or ``max_seconds`` elapse.

        Returns the concatenated PCM (int16 mono, 16 kHz) bytes. If nothing
        is heard, returns ``b""``.
        """
        pcm_chunks: list[bytes] = []
        if prepend_audio:
            pcm_chunks.append(prepend_audio[-self._preroll_bytes:])

        started = False
        silence_frames = 0
        total_frames = 0
        deadline = time.monotonic() + (self._max_frames * self._frame_ms / 1000) + 2.0
        # If the user never starts speaking within this window, bail out.
        initial_grace = time.monotonic() + 2.5

        while not stop_event.is_set():
            if time.monotonic() > deadline:
                logger.debug("VAD hit hard deadline")
                break

            frame = mic.read(timeout=0.2)
            if frame is None:
                if not started and time.monotonic() > initial_grace:
                    logger.debug("No speech detected before grace timeout")
                    break
                continue

            # webrtcvad requires exactly 10/20/30ms frames. If the mic
            # emits an odd size, feed the RMS gate instead.
            expected_bytes = int(self._sample_rate * self._frame_ms / 1000) * 2
            speaking = (
                self._is_speech(frame)
                if len(frame) == expected_bytes
                else self._rms_gate(frame)
            )

            if speaking:
                if not started:
                    logger.debug("Speech onset")
                    started = True
                silence_frames = 0
                pcm_chunks.append(frame)
                total_frames += 1
            else:
                if started:
                    pcm_chunks.append(frame)
                    total_frames += 1
                    silence_frames += 1
                    if silence_frames >= self._silence_frames_target:
                        logger.debug("Speech end detected")
                        break
                # If we haven't started yet, drop silent frames on the floor.

            if total_frames >= self._max_frames:
                logger.debug("Max command duration reached")
                break

        if not started:
            return b""
        return b"".join(pcm_chunks)
