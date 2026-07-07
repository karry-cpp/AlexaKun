"""Play a video on YouTube.

Resolution strategy (fastest / most reliable first):

1. **`yt-dlp` search** → resolve the query to a specific
   ``watch?v=<id>`` URL, then open that URL in the default browser.
   The browser (Chrome / Edge / etc.) auto-plays the video. This is
   the cleanest path — no keyboard automation, no fragile UI clicks,
   works with Hindi song titles verbatim.
2. **`pywhatkit`** fallback: opens a YouTube search page and uses
   PyAutoGUI to press play. Works but noisier.
3. **Plain search URL**: last resort — just open the results page and
   let the user click.
"""

from __future__ import annotations

import logging
import webbrowser
from urllib.parse import quote_plus
from typing import Optional

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


def _resolve_first_video_url(query: str) -> Optional[str]:
    """Use yt-dlp to find the top YouTube search result and return its
    canonical watch URL. Returns ``None`` if yt-dlp isn't installed or
    the search yields no results."""
    try:
        import yt_dlp  # local import: optional dep
    except Exception:  # noqa: BLE001
        logger.debug("yt-dlp unavailable")
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        # Avoid heavy metadata extraction; we only need the URL/ID.
        "default_search": "ytsearch1",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yt-dlp search failed for %r: %s", query, exc)
        return None

    entries = (info or {}).get("entries") or []
    if not entries:
        return None
    first = entries[0]
    # Prefer the canonical webpage URL; fall back to constructing from ID.
    url = first.get("webpage_url") or first.get("url")
    if not url:
        video_id = first.get("id")
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
    return url


def play_on_youtube(intent: Intent) -> ActionResult:
    query = intent.slots.get("query", "").strip()
    if not query:
        logger.warning("Cannot play on YouTube because query is empty.")
        return ActionResult.failure("empty query")

    logger.info("YouTube search query: %r", query)

    # Path 1: yt-dlp resolves the first video → open its watch URL.
    watch_url = _resolve_first_video_url(query)
    if watch_url:
        logger.info("Opening YouTube URL: %s", watch_url)
        webbrowser.open(watch_url, new=2)
        return ActionResult.success(
            speak_en=f"Playing {query} on YouTube.",
            speak_hi=f"YouTube pe {query} chala rahi hoon.",
        )

    # Path 2: pywhatkit legacy behaviour.
    try:
        import pywhatkit  # local import: optional dep

        pywhatkit.playonyt(query, open_video=True)
        return ActionResult.success(
            speak_en=f"Playing {query} on YouTube.",
            speak_hi=f"YouTube pe {query} chala rahi hoon.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pywhatkit failed (%s); falling back to search URL", exc)

    # Path 3: search-results URL fallback.
    fallback = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    webbrowser.open(fallback, new=2)
    return ActionResult.success(
        speak_en=f"Here are YouTube results for {query}.",
        speak_hi=f"YouTube par {query} ke results khol diye.",
    )
