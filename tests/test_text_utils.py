from __future__ import annotations

import pytest

from karry_assistant.utils.text import (
    classify_yes_no,
    contains_wake_phrase,
    detect_script,
    normalize_ascii,
    normalize_text,
    remove_wake_phrase,
    split_wake_and_command,
    strip_leading_filler,
)


WAKES = ("hey karry", "hi karry")


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Hey Karry!", "hey karry"),
            ("  hey    karry  ", "hey karry"),
            ("HEY, KARRY.", "hey karry"),
            ("", ""),
            ("play aaoge jab tum on youtube", "play aaoge jab tum on youtube"),
        ],
    )
    def test_normalize_ascii(self, raw: str, expected: str) -> None:
        assert normalize_ascii(raw) == expected

    def test_normalize_text_preserves_devanagari(self) -> None:
        # "आओगे जब तुम" (Aaoge Jab Tum in Devanagari) survives normalize_text.
        assert "आओगे" in normalize_text("आओगे जब तुम!")


class TestWakePhrase:
    @pytest.mark.parametrize(
        "utterance",
        [
            "hey karry",
            "hey karry play something",
            "HEY KARRY!",
            "hi karry, are you there",
            "hey kari lock the pc",       # minor misspelling
            "hey carry hibernate",         # very similar phrase
        ],
    )
    def test_positive(self, utterance: str) -> None:
        assert contains_wake_phrase(utterance, WAKES)

    @pytest.mark.parametrize(
        "utterance",
        [
            "",
            "just some random speech",
            "the weather is nice today",
            "tell me a joke about cats",
        ],
    )
    def test_negative(self, utterance: str) -> None:
        assert not contains_wake_phrase(utterance, WAKES)

    def test_remove_wake_phrase(self) -> None:
        assert remove_wake_phrase("hey karry play music", WAKES) == "play music"
        assert remove_wake_phrase("hey karry", WAKES) == ""

    def test_split_wake_and_command(self) -> None:
        wake, cmd = split_wake_and_command("hey karry please lock the pc", WAKES)
        assert wake is True
        assert cmd == "lock the pc"

    def test_split_no_wake(self) -> None:
        wake, cmd = split_wake_and_command("play music", WAKES)
        assert wake is False


class TestStripLeadingFiller:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("please lock the pc", "lock the pc"),
            ("can you hibernate", "hibernate"),
            ("hey ok please karry lock", "lock"),
            ("abhi lock kar do", "lock kar do"),
        ],
    )
    def test_removes_fillers(self, raw: str, expected: str) -> None:
        assert strip_leading_filler(raw) == expected


class TestYesNo:
    @pytest.mark.parametrize(
        "raw",
        [
            "yes",
            "yes please",
            "sure",
            "ok",
            "haan",
            "haan kar do",
            "theek hai",
            "go ahead",
        ],
    )
    def test_yes(self, raw: str) -> None:
        assert classify_yes_no(raw) == "yes"

    @pytest.mark.parametrize(
        "raw",
        [
            "no",
            "cancel",
            "stop",
            "nahi",
            "nahin karo",
            "mat karo",
            "abort",
        ],
    )
    def test_no(self, raw: str) -> None:
        assert classify_yes_no(raw) == "no"

    def test_unknown(self) -> None:
        assert classify_yes_no("maybe later") == "unknown"


class TestDetectScript:
    def test_english(self) -> None:
        assert detect_script("hello world") == "en"

    def test_hindi_devanagari(self) -> None:
        assert detect_script("आओगे जब तुम") == "hi"

    def test_mixed(self) -> None:
        # Any Devanagari char triggers Hindi.
        assert detect_script("play आओगे jab tum") == "hi"

    def test_hinglish_roman(self) -> None:
        # Roman-script Hinglish is classified as English (correctly).
        assert detect_script("chrome kholo") == "en"
