"""Direct answer action.

Used by the tool-calling agent for generic questions that do not need
an operating-system action: arithmetic, date differences, definitions,
short explanations, and similar Q&A. The handler deliberately does
nothing except return the answer text so the normal TTS/UI response path
can speak/display it.
"""

from __future__ import annotations

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


_MAX_ANSWER_CHARS = 500


def answer_direct(intent: Intent) -> ActionResult:
    answer = intent.slots.get("answer", "").strip()
    if not answer:
        return ActionResult.failure("empty answer")
    if len(answer) > _MAX_ANSWER_CHARS:
        answer = answer[: _MAX_ANSWER_CHARS - 3].rstrip() + "..."
    return ActionResult.success(speak_en=answer)
