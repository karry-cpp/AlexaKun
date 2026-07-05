from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Tuple

from rapidfuzz import fuzz


_NON_ALPHA_NUM = re.compile(r"[^a-z0-9\s]", flags=re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")

# Affirmative / negative tokens used for verbal confirmation (English + Hinglish).
_YES_TOKENS = frozenset(
    {
        "yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed", "ok", "okay",
        "do it", "go ahead", "proceed",
        "haan", "han", "haa", "ha", "theek hai", "thik hai", "kar do", "karo",
    }
)
_NO_TOKENS = frozenset(
    {
        "no", "nope", "nah", "cancel", "stop", "abort", "don't", "dont",
        "nahi", "nahin", "mat karo", "ruko", "ruk",
    }
)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Preserves letters
    from any script (including Devanagari vowel-sign marks like ाेु)
    so Hindi words like "आओगे जब तुम" survive intact."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    cleaned_chars = []
    for ch in text:
        if ch.isspace():
            cleaned_chars.append(" ")
            continue
        cat = unicodedata.category(ch)
        # Letters (L*), digits (N*), and combining marks (M*) all survive.
        if cat[0] in ("L", "N", "M"):
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_ascii(text: str) -> str:
    """ASCII-only normalization used for Vosk (English-only) wake matching."""
    text = text.lower().strip()
    text = _NON_ALPHA_NUM.sub(" ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def contains_wake_phrase(
    text: str,
    wake_phrases: Iterable[str],
    threshold: int = 82,
) -> bool:
    """Return True if any wake phrase is present in ``text`` (either as a
    literal substring or a fuzzy partial match at or above ``threshold``)."""
    normalized = normalize_ascii(text)
    if not normalized:
        return False
    for phrase in wake_phrases:
        normalized_phrase = normalize_ascii(phrase)
        if not normalized_phrase:
            continue
        if normalized_phrase in normalized:
            return True
        if fuzz.partial_ratio(normalized, normalized_phrase) >= threshold:
            return True
    return False


def remove_wake_phrase(text: str, wake_phrases: Iterable[str]) -> str:
    """Strip any wake phrase (and common leading fillers) from the transcript."""
    normalized = normalize_ascii(text)
    for phrase in wake_phrases:
        normalized_phrase = normalize_ascii(phrase)
        if not normalized_phrase:
            continue
        normalized = normalized.replace(normalized_phrase, " ")
    return _MULTI_SPACE.sub(" ", normalized).strip()


# Filler phrases stripped from the start of a command. Order matters:
# longer phrases are tried first so "can you please" collapses cleanly.
_LEADING_FILLERS: Tuple[str, ...] = (
    "can you please",
    "could you please",
    "would you please",
    "can you",
    "could you",
    "would you",
    "please",
    "kindly",
    "karry",
    "carry",
    "hey",
    "hi",
    "hello",
    "ok",
    "okay",
    "abhi",
    "zara",
    "jara",
)


def strip_leading_filler(text: str) -> str:
    """Remove common filler phrases from the start of a command,
    repeatedly, until nothing more can be stripped."""
    remainder = text.strip()
    changed = True
    while remainder and changed:
        changed = False
        low = remainder.lower()
        for phrase in _LEADING_FILLERS:
            if low == phrase or low.startswith(phrase + " "):
                remainder = remainder[len(phrase):].lstrip()
                changed = True
                break
    return remainder


def classify_yes_no(text: str) -> str:
    """Return ``"yes"``, ``"no"``, or ``"unknown"`` for a confirmation utterance."""
    normalized = normalize_text(text)
    if not normalized:
        return "unknown"
    tokens = normalized.split()
    # Multi-word phrases first.
    for phrase in _YES_TOKENS:
        if " " in phrase and phrase in normalized:
            return "yes"
    for phrase in _NO_TOKENS:
        if " " in phrase and phrase in normalized:
            return "no"
    # Single-token match on any word.
    for tok in tokens:
        if tok in _YES_TOKENS:
            return "yes"
        if tok in _NO_TOKENS:
            return "no"
    # Fuzzy fallback on the full utterance vs. common exemplars.
    if fuzz.partial_ratio(normalized, "yes") >= 90 or fuzz.partial_ratio(normalized, "haan") >= 90:
        return "yes"
    if fuzz.partial_ratio(normalized, "no") >= 90 or fuzz.partial_ratio(normalized, "nahi") >= 90:
        return "no"
    return "unknown"


def detect_script(text: str) -> str:
    """Return ``"hi"`` if the text contains Devanagari characters, else ``"en"``."""
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "hi"
    return "en"


def split_wake_and_command(
    text: str,
    wake_phrases: Iterable[str],
) -> Tuple[bool, str]:
    """Detect wake phrase; return ``(wake_present, command_after_wake)``."""
    wake_present = contains_wake_phrase(text, wake_phrases)
    remainder = remove_wake_phrase(text, wake_phrases) if wake_present else text
    return wake_present, strip_leading_filler(remainder)
