"""Media transport (play/pause/next/previous/stop) via Windows media keys."""

from __future__ import annotations

import ctypes

from karry_assistant.actions.registry import ActionResult
from karry_assistant.nlp.intent import Intent


_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_STOP = 0xB2
_VK_MEDIA_PLAY_PAUSE = 0xB3
_KEYEVENTF_KEYUP = 0x0002


def _press(vk_code: int) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


def play_pause(_intent: Intent) -> ActionResult:
    _press(_VK_MEDIA_PLAY_PAUSE)
    return ActionResult.success()


def next_track(_intent: Intent) -> ActionResult:
    _press(_VK_MEDIA_NEXT_TRACK)
    return ActionResult.success(speak_en="Next.", speak_hi="Agla.")


def prev_track(_intent: Intent) -> ActionResult:
    _press(_VK_MEDIA_PREV_TRACK)
    return ActionResult.success(speak_en="Previous.", speak_hi="Pichla.")


def stop(_intent: Intent) -> ActionResult:
    _press(_VK_MEDIA_STOP)
    return ActionResult.success(speak_en="Stopped.", speak_hi="Rok diya.")
