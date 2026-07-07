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
    while the user is speaking.

    Uses ``webrtcvad`` if the wheel is available. Otherwise falls back to
    an **adaptive RMS-energy gate** that:

    * Measures the noise floor from the first ``noise_calibration_ms``
      of audio, then sets the speech threshold to ``noise_floor *
      noise_multiplier`` (with a floor of ``min_threshold``). This
      adapts to how loud you actually speak, instead of guessing a
      fixed number.
    * Prepends **preroll audio** from the mic's ring buffer so the
      capture includes the moment *before* speech was first detected,
      preventing clipped first words.
    * Requires a minimum utterance length before allowing end-of-speech
      to fire, so a single quick word between two long silences doesn't
      get truncated.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        silence_ms: int = 900,
        max_seconds: float = 12.0,
        preroll_ms: int = 400,
        noise_calibration_ms: int = 300,
        noise_multiplier: float = 3.0,
        min_threshold: float = 260.0,
        min_speech_ms: int = 400,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_frames_target = max(1, silence_ms // frame_ms)
        self._max_frames = int(max_seconds * 1000 / frame_ms)
        self._preroll_bytes = int((preroll_ms / 1000) * sample_rate) * 2
        self._noise_calib_frames = max(1, noise_calibration_ms // frame_ms)
        self._noise_multiplier = noise_multiplier
        self._min_threshold = min_threshold
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)

        self._vad = None
        try:
            import webrtcvad  # local import: optional dep

            self._vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))
            logger.info("Using webrtcvad (aggressiveness=%d)", aggressiveness)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "webrtcvad unavailable (%s); using adaptive-RMS VAD "
                "with noise_multiplier=%.1f min_threshold=%.0f",
                exc,
                noise_multiplier,
                min_threshold,
            )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _rms(frame: bytes) -> float:
        """RMS amplitude of a PCM16 mono frame."""
        if not frame:
            return 0.0
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))

    def _is_speech_webrtc(self, frame: bytes) -> bool:
        expected_bytes = int(self._sample_rate * self._frame_ms / 1000) * 2
        if len(frame) != expected_bytes:
            return False
        try:
            return self._vad.is_speech(frame, self._sample_rate)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return False

    # -- public ---------------------------------------------------------
    def record(
        self,
        mic: MicStream,
        stop_event: threading.Event,
        prepend_audio: Optional[bytes] = None,
    ) -> bytes:
        """Capture speech until ``silence_ms`` of silence or ``max_seconds`` elapse.

        Returns concatenated PCM (int16 mono, 16 kHz) bytes. If nothing
        is heard, returns ``b""``.
        """
        pcm_chunks: list[bytes] = []

        # 1. Prepend preroll audio so we don't clip the first word.
        preroll = prepend_audio if prepend_audio is not None else mic.snapshot_ring()
        if preroll and self._preroll_bytes > 0:
            pcm_chunks.append(preroll[-self._preroll_bytes:])

        # 2. If webrtcvad is available, use it directly with a simple
        #    hangover loop.
        if self._vad is not None:
            return self._record_webrtc(mic, stop_event, pcm_chunks)

        # 3. Adaptive RMS path: calibrate the noise floor, then look for
        #    speech that exceeds it by a comfortable multiplier.
        noise_samples: list[float] = []
        deadline = time.monotonic() + (self._max_frames * self._frame_ms / 1000) + 2.0
        # If the user never starts speaking within this window, bail out.
        initial_grace = time.monotonic() + 3.0

        started = False
        speech_frames = 0
        silence_frames = 0
        total_frames = 0
        speech_threshold = self._min_threshold  # will be recomputed after calibration

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

            rms = self._rms(frame)

            # Calibration: first N frames feed the noise-floor estimator.
            if len(noise_samples) < self._noise_calib_frames:
                noise_samples.append(rms)
                if len(noise_samples) == self._noise_calib_frames:
                    noise_floor = float(np.median(noise_samples))
                    speech_threshold = max(
                        self._min_threshold,
                        noise_floor * self._noise_multiplier,
                    )
                    logger.debug(
                        "VAD calibration: noise=%.0f threshold=%.0f",
                        noise_floor,
                        speech_threshold,
                    )
                continue

            speaking = rms >= speech_threshold

            if speaking:
                if not started:
                    logger.debug("Speech onset at rms=%.0f (threshold=%.0f)", rms, speech_threshold)
                    started = True
                silence_frames = 0
                speech_frames += 1
                pcm_chunks.append(frame)
                total_frames += 1
            else:
                if started:
                    pcm_chunks.append(frame)
                    total_frames += 1
                    silence_frames += 1
                    # Only allow end-of-speech after a minimum utterance
                    # duration — protects against cutting off the first
                    # short word before we've heard the rest.
                    if (
                        silence_frames >= self._silence_frames_target
                        and speech_frames >= self._min_speech_frames
                    ):
                        logger.debug("Speech end detected")
                        break

            if total_frames >= self._max_frames:
                logger.debug("Max command duration reached")
                break

        if not started:
            return b""
        return b"".join(pcm_chunks)

    def _record_webrtc(
        self,
        mic: MicStream,
        stop_event: threading.Event,
        pcm_chunks: list[bytes],
    ) -> bytes:
        started = False
        silence_frames = 0
        speech_frames = 0
        total_frames = 0
        deadline = time.monotonic() + (self._max_frames * self._frame_ms / 1000) + 2.0
        initial_grace = time.monotonic() + 3.0

        while not stop_event.is_set():
            if time.monotonic() > deadline:
                break
            frame = mic.read(timeout=0.2)
            if frame is None:
                if not started and time.monotonic() > initial_grace:
                    break
                continue

            speaking = self._is_speech_webrtc(frame)
            if speaking:
                if not started:
                    logger.debug("Speech onset (webrtcvad)")
                    started = True
                silence_frames = 0
                speech_frames += 1
                pcm_chunks.append(frame)
                total_frames += 1
            else:
                if started:
                    pcm_chunks.append(frame)
                    total_frames += 1
                    silence_frames += 1
                    if (
                        silence_frames >= self._silence_frames_target
                        and speech_frames >= self._min_speech_frames
                    ):
                        break
            if total_frames >= self._max_frames:
                break

        if not started:
            return b""
        return b"".join(pcm_chunks)
