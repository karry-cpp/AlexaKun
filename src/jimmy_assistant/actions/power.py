"""Power actions: hibernate, shutdown, restart, sleep, lock.

Every command is executed via ``subprocess.run`` with a hardcoded arg
list — LLM output is only used to *choose* which of these fixed
commands runs, never to construct one. ``shutdown`` and ``restart``
include a short countdown so the user can abort with ``shutdown /a``
if they change their mind.
"""

from __future__ import annotations

import logging
import subprocess

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


CREATE_NO_WINDOW = 0x08000000  # Windows-only flag to hide the child console.


def _run(args: list[str]) -> ActionResult:
    logger.info("Running: %s", args)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        logger.error("Command failed (rc=%d): %s", completed.returncode, err)
        return ActionResult.failure(err or f"exit {completed.returncode}")
    return ActionResult.success()


def hibernate(_intent: Intent) -> ActionResult:
    result = _run(["shutdown", "/h"])
    if result.ok:
        return ActionResult.success(speak_en="Hibernating now.", speak_hi="Hibernate kar rahi hoon.")
    return result


def shutdown(_intent: Intent) -> ActionResult:
    # 5-second delay lets the user shout "cancel" (see ``shutdown /a``).
    result = _run(["shutdown", "/s", "/t", "5"])
    if result.ok:
        return ActionResult.success(
            speak_en="Shutting down in five seconds.",
            speak_hi="Paanch second mein PC band ho jayega.",
        )
    return result


def restart(_intent: Intent) -> ActionResult:
    result = _run(["shutdown", "/r", "/t", "5"])
    if result.ok:
        return ActionResult.success(
            speak_en="Restarting in five seconds.",
            speak_hi="Paanch second mein restart ho raha hai.",
        )
    return result


def sleep(_intent: Intent) -> ActionResult:
    # SetSuspendState(bHibernate=0, bForce=1, bWakeupEventsDisabled=0)
    result = _run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
    if result.ok:
        return ActionResult.success(speak_en="Going to sleep.", speak_hi="So rahi hoon.")
    return result


def lock(_intent: Intent) -> ActionResult:
    result = _run(["rundll32.exe", "user32.dll,LockWorkStation"])
    if result.ok:
        return ActionResult.success(speak_en="Locked.", speak_hi="Lock kar diya.")
    return result


def cancel_pending_shutdown() -> ActionResult:
    """Used by the 'cancel' intent to abort a queued shutdown/restart."""
    return _run(["shutdown", "/a"])
