from __future__ import annotations

from typing import Any, Dict, List

from jimmy_assistant.actions.registry import ActionRegistry, ActionResult, ToolSchema
from jimmy_assistant.nlp.agent import Agent
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.ollama_client import ChatResponse, ToolCall
from jimmy_assistant.ui.events import JimmyListener, NullListener


class _RecordingListener(JimmyListener):
    def __init__(self) -> None:
        self.events: List[tuple] = []

    def on_status(self, message: str) -> None:
        self.events.append(("status", message))

    def on_heard(self, text: str, language: str) -> None:
        self.events.append(("heard", text, language))

    def on_tool_call(self, tool: str, arguments: Dict[str, Any]) -> None:
        self.events.append(("tool_call", tool, dict(arguments)))

    def on_tool_result(self, tool: str, ok: bool, summary: str) -> None:
        self.events.append(("tool_result", tool, ok, summary))

    def on_response(self, text: str) -> None:
        self.events.append(("response", text))

    def on_error(self, text: str) -> None:
        self.events.append(("error", text))

    def on_confirm_prompt(self, tool: str, prompt: str) -> None:
        self.events.append(("confirm_prompt", tool, prompt))

    def on_confirm_answer(self, answer: str) -> None:
        self.events.append(("confirm_answer", answer))


class _StubLLM:
    def __init__(self, script: List[ChatResponse]) -> None:
        self._script = list(script)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> ChatResponse:
        if not self._script:
            return ChatResponse(content="")
        return self._script.pop(0)


def _ok(intent: Intent) -> ActionResult:
    return ActionResult.success(speak_en=f"did {intent.name}")


def _fail(_intent: Intent) -> ActionResult:
    return ActionResult.failure("broken")


def _make_registry() -> ActionRegistry:
    r = ActionRegistry()
    r.register(
        "web.search",
        _ok,
        schema=ToolSchema(
            description="web search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
    )
    r.register(
        "power.shutdown",
        _ok,
        destructive=True,
        schema=ToolSchema(description="shutdown", parameters={"type": "object", "properties": {}, "required": []}),
    )
    r.register(
        "media.stop",
        _fail,
        schema=ToolSchema(description="stop", parameters={"type": "object", "properties": {}, "required": []}),
    )
    return r


class TestAgentPublishesEvents:
    def test_tool_call_and_result(self) -> None:
        llm = _StubLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="web.search", arguments={"query": "cats"})]),
                ChatResponse(content="Searching for cats."),
            ]
        )
        listener = _RecordingListener()
        agent = Agent(registry=_make_registry(), llm=llm, listener=listener)
        agent.run("google cats")

        kinds = [e[0] for e in listener.events]
        assert "tool_call" in kinds
        assert "tool_result" in kinds

        tc = next(e for e in listener.events if e[0] == "tool_call")
        tr = next(e for e in listener.events if e[0] == "tool_result")
        assert tc[1] == "web.search"
        assert tc[2] == {"query": "cats"}
        assert tr[1] == "web.search"
        assert tr[2] is True

    def test_destructive_denied_still_publishes(self) -> None:
        llm = _StubLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="power.shutdown", arguments={})]),
                ChatResponse(content="ok cancelled"),
            ]
        )
        listener = _RecordingListener()
        agent = Agent(
            registry=_make_registry(),
            llm=llm,
            confirm_cb=lambda _intent: False,
            listener=listener,
        )
        agent.run("shutdown now")

        tr = next(e for e in listener.events if e[0] == "tool_result")
        assert tr[2] is False
        assert "did not confirm" in tr[3]

    def test_handler_failure_reported(self) -> None:
        llm = _StubLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="media.stop", arguments={})]),
                ChatResponse(content="that failed"),
            ]
        )
        listener = _RecordingListener()
        agent = Agent(registry=_make_registry(), llm=llm, listener=listener)
        agent.run("stop")

        tr = next(e for e in listener.events if e[0] == "tool_result")
        assert tr[2] is False
        assert "broken" in tr[3]

    def test_unknown_tool_reported(self) -> None:
        llm = _StubLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="does.not.exist", arguments={})]),
                ChatResponse(content="oh well"),
            ]
        )
        listener = _RecordingListener()
        agent = Agent(registry=_make_registry(), llm=llm, listener=listener)
        agent.run("impossible")

        tr = next(e for e in listener.events if e[0] == "tool_result")
        assert tr[2] is False
        assert "unknown tool" in tr[3]


class TestNullListenerDefault:
    """The agent must accept a missing listener without crashing."""

    def test_no_listener(self) -> None:
        llm = _StubLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="web.search", arguments={"query": "cats"})]),
                ChatResponse(content="done"),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm)  # no listener kw
        outcome = agent.run("google cats")
        assert outcome.steps[0].result_ok is True


class TestListenerProtocol:
    def test_null_listener_matches_protocol(self) -> None:
        listener: JimmyListener = NullListener()
        listener.on_status("hello")
        listener.on_heard("hi", "en")
        listener.on_tool_call("t", {"a": 1})
        listener.on_tool_result("t", True, "ok")
        listener.on_response("bye")
        listener.on_error("boom")
        listener.on_confirm_prompt("t", "sure?")
        listener.on_confirm_answer("yes")
