from __future__ import annotations

from jimmy_assistant.nlp import intent as A
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.router import IntentRouter
from jimmy_assistant.nlp.rules import RulesParser


class _FakeLLM:
    def __init__(self, response: Intent) -> None:
        self._response = response
        self.calls = 0

    def resolve(self, transcript: str) -> Intent:
        self.calls += 1
        return Intent(
            name=self._response.name,
            slots=dict(self._response.slots),
            source="llm",
            raw_text=transcript,
        )


class TestRouter:
    def test_rules_wins(self) -> None:
        llm = _FakeLLM(Intent(name=A.ACTION_UNKNOWN))
        router = IntentRouter(rules=RulesParser(), llm=llm, llm_enabled=True)
        result = router.resolve("lock the pc")
        assert result.name == A.ACTION_LOCK
        assert result.source == "rules"
        assert llm.calls == 0

    def test_falls_back_to_llm(self) -> None:
        llm = _FakeLLM(
            Intent(name=A.ACTION_YOUTUBE_PLAY, slots={"query": "some song"}, source="llm")
        )
        router = IntentRouter(rules=RulesParser(), llm=llm, llm_enabled=True)
        # Deliberately weird phrasing that no rule handles.
        result = router.resolve("ummm can you please put on that one song for me")
        assert llm.calls == 1
        assert result.name == A.ACTION_YOUTUBE_PLAY
        assert result.source == "llm"

    def test_llm_disabled(self) -> None:
        llm = _FakeLLM(Intent(name=A.ACTION_YOUTUBE_PLAY, slots={"query": "x"}))
        router = IntentRouter(rules=RulesParser(), llm=llm, llm_enabled=False)
        result = router.resolve("do the thing")
        assert llm.calls == 0
        assert result.is_unknown

    def test_empty_text(self) -> None:
        router = IntentRouter(rules=RulesParser(), llm=None, llm_enabled=False)
        assert router.resolve("").is_unknown
        assert router.resolve("   ").is_unknown
