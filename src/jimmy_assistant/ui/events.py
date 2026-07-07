"""Event-listener interface for Jimmy.

The orchestrator publishes lifecycle events (wake detected, transcript
received, tool called, response spoken, error raised, etc.) through a
:class:`JimmyListener`. Any UI — Tkinter window, tray tooltip, or a
future web dashboard — can implement the protocol and subscribe.

Listeners are *always* invoked from Jimmy's worker threads (audio and
LLM loops), never on the UI thread. UI implementations must marshal
these calls onto their main thread (e.g. Tk ``.after`` or Qt signals).

The default :class:`NullListener` swallows all events so calling code
never has to check for ``None``.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class JimmyListener(Protocol):
    """Observer interface for the Jimmy orchestrator.

    Every method has a default no-op implementation via
    :class:`NullListener`, so a UI only needs to override the events
    it cares about.
    """

    def on_status(self, message: str) -> None: ...
    def on_heard(self, text: str, language: str) -> None: ...
    def on_tool_call(self, tool: str, arguments: Dict[str, Any]) -> None: ...
    def on_tool_result(self, tool: str, ok: bool, summary: str) -> None: ...
    def on_response(self, text: str) -> None: ...
    def on_error(self, text: str) -> None: ...
    def on_confirm_prompt(self, tool: str, prompt: str) -> None: ...
    def on_confirm_answer(self, answer: str) -> None: ...


class NullListener:
    """No-op implementation used as the default when no UI is attached."""

    def on_status(self, message: str) -> None:  # noqa: D401
        return None

    def on_heard(self, text: str, language: str) -> None:
        return None

    def on_tool_call(self, tool: str, arguments: Dict[str, Any]) -> None:
        return None

    def on_tool_result(self, tool: str, ok: bool, summary: str) -> None:
        return None

    def on_response(self, text: str) -> None:
        return None

    def on_error(self, text: str) -> None:
        return None

    def on_confirm_prompt(self, tool: str, prompt: str) -> None:
        return None

    def on_confirm_answer(self, answer: str) -> None:
        return None
