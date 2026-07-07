from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_USAGE = """\
Jimmy — voice assistant.

Usage:
  python run.py              # system tray icon (default)
  python run.py --window     # transcript window (recommended for demos)
  python run.py --console    # headless console
  python run.py --help
"""


def _run() -> int:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return 0
    if "--console" in argv:
        from jimmy_assistant.main import run

        run()
        return 0
    if "--window" in argv:
        from jimmy_assistant.ui.window import main as window_main

        return window_main()

    from jimmy_assistant.ui.tray import main as tray_main

    return tray_main()


if __name__ == "__main__":
    raise SystemExit(_run())
