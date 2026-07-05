from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from karry_assistant.nlp import intent as A
from karry_assistant.nlp.intent import Intent
from karry_assistant.nlp.prompts import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat / tool-calling types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the LLM asked us to perform."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    """Structured response from Ollama's ``/api/chat`` endpoint."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class OllamaClient:
    """Client for the local Ollama server.

    Two modes:

    1. :meth:`resolve` — legacy single-shot JSON-mode intent extraction.
       Kept for backwards compatibility with the old :class:`IntentRouter`.
    2. :meth:`chat` — OpenAI-style multi-turn chat with tool calling,
       used by :class:`karry_assistant.nlp.agent.Agent` for full
       agentic behaviour.

    Both are non-throwing: any transport / decoding failure returns a
    safe fallback (unknown intent, empty tool_calls, or ``error`` set).
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:3b-instruct",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    # -- diagnostics ----------------------------------------------------
    def is_reachable(self) -> bool:
        try:
            r = httpx.get(f"{self._url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # -- legacy single-shot JSON-mode ----------------------------------
    def resolve(self, transcript: str) -> Intent:
        if not transcript.strip():
            return Intent(name=A.ACTION_UNKNOWN, source="llm", raw_text=transcript)

        payload = {
            "model": self._model,
            "system": SYSTEM_PROMPT,
            "prompt": f'user: "{transcript}"\n',
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 200,
            },
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(f"{self._url}/api/generate", json=payload)
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as exc:
            logger.warning("Ollama request failed: %s", exc)
            return Intent(name=A.ACTION_UNKNOWN, source="llm", raw_text=transcript)

        raw_response = (body or {}).get("response", "").strip()
        parsed = self._parse_single_intent(raw_response)
        if parsed is None:
            logger.warning("Ollama returned unparseable JSON: %r", raw_response)
            return Intent(name=A.ACTION_UNKNOWN, source="llm", raw_text=transcript)

        return Intent(
            name=parsed.name,
            slots=parsed.slots,
            source="llm",
            raw_text=transcript,
        )

    def _parse_single_intent(self, raw: str) -> Optional[Intent]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        action = str(data.get("action", "")).strip()
        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if action not in A.KNOWN_ACTIONS and action != A.ACTION_UNKNOWN:
            logger.warning("Ollama proposed unknown action %r", action)
            return Intent(name=A.ACTION_UNKNOWN)

        slots: Dict[str, str] = {}
        for k, v in params.items():
            if v is None:
                continue
            s = str(v).strip()
            if s:
                slots[str(k)] = s

        return Intent(name=action, slots=slots)

    # -- tool-calling chat (agent mode) --------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Call ``/api/chat`` with an optional ``tools`` array (OpenAI
        function-calling schema). Returns the assistant message
        content and/or any tool calls the model requested.
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                # Tool-calling picks a single function name + small
                # argument object — 200 tokens is more than enough and
                # keeps LLM latency low on CPU-bound Ollama models.
                "num_predict": 200,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(f"{self._url}/api/chat", json=payload)
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as exc:
            logger.warning("Ollama chat request failed: %s", exc)
            return ChatResponse(error=str(exc))

        message = (body or {}).get("message", {}) or {}
        content = (message.get("content") or "").strip()
        raw_calls = message.get("tool_calls") or []

        parsed_calls: List[ToolCall] = []
        for call in raw_calls:
            func = (call or {}).get("function") or {}
            name = str(func.get("name") or "").strip()
            if not name:
                continue
            args = func.get("arguments") or {}
            # Ollama sometimes returns arguments as a JSON string; be
            # lenient about that.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            parsed_calls.append(ToolCall(name=name, arguments=args))

        return ChatResponse(content=content, tool_calls=parsed_calls)
