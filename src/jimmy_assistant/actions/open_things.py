"""Open URLs, local files, or folders."""

from __future__ import annotations

import logging
import os
import re
import webbrowser
from pathlib import Path

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def open_thing(intent: Intent) -> ActionResult:
    target = intent.slots.get("target", "").strip()
    if not target:
        return ActionResult.failure("open target is empty")

    # URL branch --------------------------------------------------------
    if _URL_PATTERN.match(target):
        logger.info("Opening URL: %s", target)
        webbrowser.open(target, new=2)
        return ActionResult.success(
            speak_en="Opening the link.",
            speak_hi="Link khol rahi hoon.",
        )

    # Filesystem branch -------------------------------------------------
    candidate = Path(os.path.expandvars(os.path.expanduser(target))).resolve()
    if not candidate.exists():
        return ActionResult.failure(f"path not found: {candidate}")

    try:
        os.startfile(str(candidate))
    except OSError as exc:
        return ActionResult.failure(f"could not open: {exc}")

    kind = "folder" if candidate.is_dir() else "file"
    return ActionResult.success(
        speak_en=f"Opening the {kind}.",
        speak_hi=f"{kind} khol rahi hoon.",
    )
