from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from karry_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


ActionHandler = Callable[[Intent], "ActionResult"]


@dataclass(frozen=True)
class ActionResult:
    """What a handler reports back after (attempting to) execute."""

    ok: bool
    speak_en: str = ""      # optional confirmation phrasing
    speak_hi: str = ""
    error: str = ""

    @classmethod
    def success(cls, speak_en: str = "", speak_hi: str = "") -> "ActionResult":
        return cls(ok=True, speak_en=speak_en, speak_hi=speak_hi)

    @classmethod
    def failure(cls, error: str) -> "ActionResult":
        return cls(ok=False, error=error)


@dataclass(frozen=True)
class ToolSchema:
    """OpenAI-style JSON schema describing a callable tool.

    Handed to the LLM so it can pick the right tool and fill in
    parameters. Compatible with Ollama ``/api/chat`` ``tools`` field.
    """

    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    slot_map: Dict[str, str] = field(default_factory=dict)    # llm_param -> intent_slot

    def to_openai(self, name: str) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


@dataclass(frozen=True)
class _RegisteredAction:
    handler: ActionHandler
    is_destructive: bool = False
    confirm_prompt_en: str = "Are you sure?"
    confirm_prompt_hi: str = "kya aap sure ho?"
    schema: Optional[ToolSchema] = None


class ActionRegistry:
    """Central dispatcher: maps canonical action names to handler functions
    and enforces which actions require verbal confirmation before executing.

    **Safety invariant**: LLM output determines *which* handler runs, but
    the handler itself constructs the actual OS command with hardcoded
    argument lists. Free-text slot values are only used as *parameters*
    (e.g. a YouTube search query), never as shell commands.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, _RegisteredAction] = {}

    def register(
        self,
        name: str,
        handler: ActionHandler,
        *,
        destructive: bool = False,
        confirm_en: str = "Are you sure?",
        confirm_hi: str = "kya aap sure ho?",
        schema: Optional[ToolSchema] = None,
    ) -> None:
        if name in self._actions:
            logger.warning("Overwriting existing handler for %s", name)
        self._actions[name] = _RegisteredAction(
            handler=handler,
            is_destructive=destructive,
            confirm_prompt_en=confirm_en,
            confirm_prompt_hi=confirm_hi,
            schema=schema,
        )

    def has(self, name: str) -> bool:
        return name in self._actions

    def names(self) -> List[str]:
        return list(self._actions.keys())

    def is_destructive(self, name: str) -> bool:
        entry = self._actions.get(name)
        return bool(entry and entry.is_destructive)

    def confirm_prompt(self, name: str, lang: str = "en") -> str:
        entry = self._actions.get(name)
        if not entry:
            return "Are you sure?"
        return entry.confirm_prompt_hi if lang == "hi" else entry.confirm_prompt_en

    def openai_tools(self) -> List[Dict[str, Any]]:
        """Emit all registered tools as an OpenAI-compatible ``tools`` array."""
        tools: List[Dict[str, Any]] = []
        for name, entry in self._actions.items():
            if entry.schema is None:
                # Skip tools without an explicit schema — they can still be
                # dispatched from the rules parser, but the LLM should not
                # be able to pick them.
                continue
            tools.append(entry.schema.to_openai(name))
        return tools

    def build_intent(self, name: str, arguments: Dict[str, Any]) -> Intent:
        """Convert LLM-provided ``arguments`` into an :class:`Intent`
        with slot names the handler expects."""
        entry = self._actions.get(name)
        slot_map = (entry.schema.slot_map if entry and entry.schema else {}) or {}
        slots: Dict[str, str] = {}
        for llm_key, value in (arguments or {}).items():
            slot_key = slot_map.get(llm_key, llm_key)
            if value is None:
                continue
            slots[slot_key] = str(value)
        return Intent(name=name, slots=slots, source="llm")

    def dispatch(self, intent: Intent) -> ActionResult:
        entry = self._actions.get(intent.name)
        if entry is None:
            logger.warning("No handler for action %s", intent.name)
            return ActionResult.failure(f"no handler for {intent.name}")
        try:
            logger.info("Dispatching %s slots=%s", intent.name, intent.slots)
            result = entry.handler(intent)
            if not isinstance(result, ActionResult):
                # Handlers may return None for backwards compatibility.
                return ActionResult.success()
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Handler for %s raised", intent.name)
            return ActionResult.failure(str(exc))

    # Backwards-compatibility with the earlier ``execute`` name.
    def execute(self, intent: Intent) -> bool:
        return self.dispatch(intent).ok
