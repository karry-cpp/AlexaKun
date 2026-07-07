"""Tkinter transcript window for Jimmy.

Shows a live conversation view: the text Jimmy heard, the tools it
called, and the responses it spoke. Runs the ``Jimmy`` orchestrator
in a background thread; Tk's mainloop owns the main thread.

Design notes
------------
* All Tk widget updates go through :meth:`_dispatch` on the main
  thread. Jimmy publishes events from worker threads, so we route them
  into a :class:`queue.Queue` and drain it on a periodic ``after``
  callback.
* The window is intentionally minimal — a scrolling text area with
  colored role prefixes and a small status bar. Adding fancy styling
  later is easy without touching the event-plumbing.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
from typing import Any, Dict

from jimmy_assistant.config import load_settings
from jimmy_assistant.main import Jimmy
from jimmy_assistant.ui.events import JimmyListener
from jimmy_assistant.utils.logging import configure_logging


class _QueueListener(JimmyListener):
    """:class:`JimmyListener` that posts every event onto a queue.

    Jimmy publishes events from worker threads; the window drains
    the queue on the Tk main thread via ``after``.
    """

    def __init__(self, q: "queue.Queue[tuple]") -> None:
        self._q = q

    def on_status(self, message: str) -> None:
        self._q.put(("status", message))

    def on_heard(self, text: str, language: str) -> None:
        self._q.put(("heard", text, language))

    def on_tool_call(self, tool: str, arguments: Dict[str, Any]) -> None:
        self._q.put(("tool_call", tool, arguments))

    def on_tool_result(self, tool: str, ok: bool, summary: str) -> None:
        self._q.put(("tool_result", tool, ok, summary))

    def on_response(self, text: str) -> None:
        self._q.put(("response", text))

    def on_error(self, text: str) -> None:
        self._q.put(("error", text))

    def on_confirm_prompt(self, tool: str, prompt: str) -> None:
        self._q.put(("confirm_prompt", tool, prompt))

    def on_confirm_answer(self, answer: str) -> None:
        self._q.put(("confirm_answer", answer))


class TranscriptWindow:
    """Main Tk window showing what Jimmy hears and how it responds."""

    _COLORS = {
        "user": "#3B82F6",       # blue
        "assistant": "#16A34A",   # green
        "tool": "#7C3AED",        # purple
        "error": "#DC2626",       # red
        "status": "#6B7280",      # gray
        "confirm": "#D97706",     # amber
    }

    def __init__(self) -> None:
        self._settings = load_settings()
        configure_logging(self._settings.log_level)

        self._event_q: "queue.Queue[tuple]" = queue.Queue()
        listener = _QueueListener(self._event_q)
        self._jimmy = Jimmy(self._settings, listener=listener)
        self._worker: threading.Thread | None = None

        self._root = tk.Tk()
        self._root.title("Jimmy")
        self._root.geometry("640x520")
        self._root.minsize(420, 320)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    # -- UI construction ------------------------------------------------
    def _build_ui(self) -> None:
        # Status bar (top).
        self._status_var = tk.StringVar(value="starting…")
        status = tk.Label(
            self._root,
            textvariable=self._status_var,
            anchor="w",
            padx=10,
            pady=6,
            fg="white",
            bg="#111827",
            font=("Segoe UI", 10),
        )
        status.pack(side="top", fill="x")

        # Transcript (middle).
        self._text = scrolledtext.ScrolledText(
            self._root,
            wrap="word",
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
            bg="#F9FAFB",
            fg="#111827",
            state="disabled",
        )
        self._text.pack(side="top", fill="both", expand=True)

        # Configure role tags for colored prefixes.
        for name, color in self._COLORS.items():
            self._text.tag_configure(name, foreground=color, font=("Segoe UI", 11, "bold"))

        # Footer buttons.
        footer = tk.Frame(self._root, padx=10, pady=8)
        footer.pack(side="bottom", fill="x")

        tk.Button(footer, text="Clear", command=self._clear).pack(side="left")
        tk.Button(footer, text="Quit", command=self._on_close).pack(side="right")

        self._hint = tk.Label(
            footer,
            text=f"Say '{self._settings.wake_phrases[0]}' to talk to Jimmy.",
            fg="#6B7280",
            font=("Segoe UI", 9, "italic"),
        )
        self._hint.pack(side="left", padx=10)

    def _append(self, role_tag: str, prefix: str, body: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", f"{prefix}: ", role_tag)
        self._text.insert("end", body + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def _clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    # -- event dispatch (Tk main thread) -------------------------------
    def _poll_events(self) -> None:
        while True:
            try:
                evt = self._event_q.get_nowait()
            except queue.Empty:
                break
            self._render(evt)
        self._root.after(60, self._poll_events)

    def _render(self, evt: tuple) -> None:
        kind = evt[0]
        if kind == "status":
            self._status_var.set(evt[1])
        elif kind == "heard":
            text, lang = evt[1], evt[2]
            self._append("user", f"You ({lang})", text)
        elif kind == "tool_call":
            tool, args = evt[1], evt[2]
            self._append("tool", "Tool", f"{tool}({args})")
        elif kind == "tool_result":
            tool, ok, summary = evt[1], evt[2], evt[3]
            tag = "tool" if ok else "error"
            self._append(tag, "Result", f"{tool} → {'OK' if ok else 'FAIL'}: {summary}")
        elif kind == "response":
            self._append("assistant", "Jimmy", evt[1])
        elif kind == "error":
            self._append("error", "Error", evt[1])
        elif kind == "confirm_prompt":
            tool, prompt = evt[1], evt[2]
            self._append("confirm", "Confirm?", f"{tool}: {prompt}")
        elif kind == "confirm_answer":
            self._append("confirm", "Answer", evt[1])

    # -- lifecycle ------------------------------------------------------
    def _start_worker(self) -> None:
        def _target() -> None:
            try:
                self._jimmy.run()
            except Exception as exc:  # noqa: BLE001
                self._event_q.put(("error", f"Jimmy crashed: {exc}"))

        self._worker = threading.Thread(target=_target, name="jimmy-main", daemon=True)
        self._worker.start()

    def _on_close(self) -> None:
        self._jimmy.request_stop()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        self._root.destroy()

    def run(self) -> int:
        self._start_worker()
        self._root.after(60, self._poll_events)
        self._root.mainloop()
        return 0


def main() -> int:
    window = TranscriptWindow()
    return window.run()


if __name__ == "__main__":
    raise SystemExit(main())
