from __future__ import annotations

from karry_assistant.actions.registry import ActionRegistry, ActionResult
from karry_assistant.nlp.intent import Intent


def _ok_handler(intent: Intent) -> ActionResult:
    return ActionResult.success(speak_en="done")


def _fail_handler(_intent: Intent) -> ActionResult:
    return ActionResult.failure("boom")


def _raise_handler(_intent: Intent) -> ActionResult:
    raise RuntimeError("kaboom")


class TestRegistry:
    def test_dispatch_ok(self) -> None:
        r = ActionRegistry()
        r.register("test.ok", _ok_handler)
        result = r.dispatch(Intent(name="test.ok"))
        assert result.ok is True
        assert result.speak_en == "done"

    def test_dispatch_failure(self) -> None:
        r = ActionRegistry()
        r.register("test.fail", _fail_handler)
        result = r.dispatch(Intent(name="test.fail"))
        assert result.ok is False
        assert "boom" in result.error

    def test_dispatch_raises(self) -> None:
        r = ActionRegistry()
        r.register("test.raise", _raise_handler)
        result = r.dispatch(Intent(name="test.raise"))
        assert result.ok is False
        assert "kaboom" in result.error

    def test_missing_handler(self) -> None:
        r = ActionRegistry()
        result = r.dispatch(Intent(name="nonexistent"))
        assert result.ok is False
        assert "no handler" in result.error

    def test_destructive_flag(self) -> None:
        r = ActionRegistry()
        r.register("test.safe", _ok_handler)
        r.register(
            "test.danger",
            _ok_handler,
            destructive=True,
            confirm_en="Sure?",
            confirm_hi="Pakka?",
        )

        assert r.is_destructive("test.safe") is False
        assert r.is_destructive("test.danger") is True
        assert r.is_destructive("test.unknown") is False

        assert r.confirm_prompt("test.danger", "en") == "Sure?"
        assert r.confirm_prompt("test.danger", "hi") == "Pakka?"

    def test_has(self) -> None:
        r = ActionRegistry()
        r.register("x", _ok_handler)
        assert r.has("x")
        assert not r.has("y")
