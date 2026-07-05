"""System-tray icon for Karry.

Runs the assistant in a worker thread and exposes control via a
minimal pystray menu (Pause / Resume / Open logs / Quit). Falls back to
console-only mode if ``pystray`` isn't installed — the assistant is
useful either way.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

from karry_assistant.config import load_settings
from karry_assistant.main import Karry
from karry_assistant.utils.logging import configure_logging, logs_dir


logger = logging.getLogger("karry.tray")


def _make_icon_image():
    """Produce a simple in-memory PIL image so we don't need to ship a PNG."""
    try:
        from PIL import Image, ImageDraw  # local import: optional dep
    except Exception:  # noqa: BLE001
        return None

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Simple filled circle with a "K".
    draw.ellipse((4, 4, 60, 60), fill=(30, 30, 30, 255))
    draw.text((21, 15), "K", fill=(255, 255, 255, 255))
    return img


class KarryTray:
    def __init__(self) -> None:
        self._settings = load_settings()
        configure_logging(self._settings.log_level)
        self._karry: Optional[Karry] = None
        self._worker: Optional[threading.Thread] = None
        self._paused = False

    # -- lifecycle ------------------------------------------------------
    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._karry = Karry(self._settings)

        def _target() -> None:
            try:
                assert self._karry is not None
                self._karry.run()
            except Exception:  # noqa: BLE001
                logger.exception("Karry worker crashed")

        self._worker = threading.Thread(target=_target, name="karry-main", daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._karry is not None:
            self._karry.request_stop()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        self._karry = None
        self._worker = None

    # -- pystray callbacks ---------------------------------------------
    def _on_quit(self, icon: object, _item: object) -> None:
        logger.info("Tray: quit")
        self._stop_worker()
        try:
            icon.stop()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _on_toggle_pause(self, icon: object, _item: object) -> None:
        if self._paused:
            self._start_worker()
            self._paused = False
        else:
            self._stop_worker()
            self._paused = True
        try:
            icon.update_menu()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _on_open_logs(self, _icon: object, _item: object) -> None:
        try:
            os.startfile(str(logs_dir()))
        except OSError:
            logger.exception("Could not open logs dir")

    # -- run ------------------------------------------------------------
    def run(self) -> None:
        try:
            import pystray  # local import: optional dep
        except Exception:  # noqa: BLE001
            logger.warning("pystray unavailable — running in console mode")
            self._start_worker()
            try:
                assert self._worker is not None
                self._worker.join()
            except KeyboardInterrupt:
                self._stop_worker()
            return

        icon_image = _make_icon_image()
        if icon_image is None:
            logger.warning("Pillow unavailable — running in console mode")
            self._start_worker()
            try:
                assert self._worker is not None
                self._worker.join()
            except KeyboardInterrupt:
                self._stop_worker()
            return

        def _pause_label(_item: object) -> str:
            return "Resume" if self._paused else "Pause"

        menu = pystray.Menu(
            pystray.MenuItem(_pause_label, self._on_toggle_pause),
            pystray.MenuItem("Open logs", self._on_open_logs),
            pystray.MenuItem("Quit", self._on_quit),
        )

        self._start_worker()
        icon = pystray.Icon("karry", icon_image, "Karry", menu)
        icon.run()


def main() -> int:
    tray = KarryTray()
    tray.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
