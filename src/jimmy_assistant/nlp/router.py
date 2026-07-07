from __future__ import annotations

import logging
from typing import Optional

from jimmy_assistant.nlp import intent as A
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.ollama_client import OllamaClient
from jimmy_assistant.nlp.rules import RulesParser


logger = logging.getLogger(__name__)


class IntentRouter:
    """Resolve a natural-language command into a canonical :class:`Intent`.

    Strategy: rule-based fast path first, then Ollama LLM fallback.
    Rules always win when they match — they are faster and more
    predictable. The LLM is only consulted for out-of-vocabulary
    phrasing so it doesn't slow down the common case.
    """

    def __init__(
        self,
        rules: Optional[RulesParser] = None,
        llm: Optional[OllamaClient] = None,
        llm_enabled: bool = True,
    ) -> None:
        self._rules = rules or RulesParser()
        self._llm = llm
        self._llm_enabled = llm_enabled and (llm is not None)

    def resolve(self, transcript: str) -> Intent:
        transcript = transcript.strip()
        if not transcript:
            return Intent(name=A.ACTION_UNKNOWN, raw_text="")

        # 1. Try regex fast-path.
        rule_hit = self._rules.parse(transcript)
        if rule_hit is not None and not rule_hit.is_unknown:
            logger.debug("Router: rules matched %s", rule_hit.name)
            return rule_hit

        # 2. LLM fallback (if configured and reachable).
        if self._llm_enabled and self._llm is not None:
            logger.debug("Router: escalating to LLM")
            llm_hit = self._llm.resolve(transcript)
            if not llm_hit.is_unknown:
                return llm_hit

        logger.info("Router: no match for %r", transcript)
        return Intent(name=A.ACTION_UNKNOWN, raw_text=transcript)
