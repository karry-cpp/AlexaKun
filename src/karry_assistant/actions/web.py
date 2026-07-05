"""Simple web-search action: opens the default browser at Google."""

from __future__ import annotations

import logging
import webbrowser
from urllib.parse import quote_plus

from karry_assistant.actions.registry import ActionResult
from karry_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


def web_search(intent: Intent) -> ActionResult:
    query = intent.slots.get("query", "").strip()
    if not query:
        return ActionResult.failure("empty search query")
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    logger.info("Opening web search: %s", url)
    webbrowser.open(url, new=2)
    return ActionResult.success(
        speak_en=f"Searching for {query}.",
        speak_hi=f"{query} search kar rahi hoon.",
    )
