from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Deque, Optional

import sounddevice as sd


logger = logging.getLogger(__name__)


class MicStream:
    """A single shared microphone stream that produces 16-bit mono PCM
    frames. Consumers (wake-word detector, VAD recorder) pull frames via
    :meth:`read`. The stream also maintains a short rolling ring buffer of
    recent audio so callers can prepend a few hundred ms of context to a
    capture (useful when users say the wake phrase and command in one breath).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
        frame_ms: int = 30,
        ring_seconds: float = 1.5,
    ) -> None:
        self._sample_rate = sample_rate
        self._device_index = device_index
        self._frame_samples = int(sample_rate * frame_ms / 1000)  # 480 at 16k/30ms
        self._frame_bytes = self._frame_samples * 2               # int16 mono

        self._q: "queue.Queue[bytes]" = queue.Queue()
        max_ring = int((ring_seconds * 1000) / frame_ms)
        self._ring: Deque[bytes] = deque(maxlen=max_ring)
        self._ring_lock = threading.Lock()

        self._stream: Optional[sd.RawInputStream] = None

    # -- properties ------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    # -- lifecycle -------------------------------------------------------
    def _callback(self, indata: bytes, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            logger.debug("Mic callback status: %s", status)
        chunk = bytes(indata)
        self._q.put(chunk)
        with self._ring_lock:
            self._ring.append(chunk)

    def __enter__(self) -> "MicStream":
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_samples,
            device=self._device_index,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Microphone stream started (rate=%d, device=%s, frame=%dms)",
            self._sample_rate,
            self._device_index,
            int(self._frame_samples * 1000 / self._sample_rate),
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing mic stream")
        self._stream = None

    # -- reads -----------------------------------------------------------
    def read(self, timeout: float = 0.25) -> Optional[bytes]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def snapshot_ring(self) -> bytes:
        """Return the last ``ring_seconds`` of audio as raw PCM bytes."""
        with self._ring_lock:
            return b"".join(self._ring)
