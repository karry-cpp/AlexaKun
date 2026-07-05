"""Karry orchestrator.

High-level control flow:

    ┌─────────────────────────────────────────────────────────────┐
    │  mic stream (16 kHz mono, 30 ms frames)                     │
    └───────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────▼──────────┐
        │ VoskWakeDetector      │  Vosk small-en model, "hey karry"
        └───────────┬──────────┘
                    │ (wake event)
        ┌───────────▼──────────┐
        │ VadCommandRecorder    │  webrtcvad, records until silence
        └───────────┬──────────┘
                    │ (PCM bytes)
        ┌───────────▼──────────┐
        │ WhisperSTT            │  faster-whisper large-v3 (Hindi + accent)
        └───────────┬──────────┘
                    │ (text, lang)
        ┌───────────▼──────────┐
        │ IntentRouter          │  rules → Ollama qwen2.5:3b JSON fallback
        └───────────┬──────────┘
                    │ (Intent)
        ┌───────────▼──────────┐
        │ Confirmation gate     │  spoken "yes/no" for destructive actions
        └───────────┬──────────┘
                    │ (Intent)
        ┌───────────▼──────────┐
        │ ActionRegistry        │  runs the whitelisted handler
        └───────────┬──────────┘
                    │ (ActionResult)
        ┌───────────▼──────────┐
        │ Speaker               │  TTS confirmation → back to waiting for wake
        └──────────────────────┘

If the user says everything in one breath ("hey karry lock the pc")
Vosk returns the whole utterance and, when the residual matches a rule,
we skip re-capture and Whisper for that turn. Anything not recognized
by rules is always re-captured via Whisper so accented English and
Hindi words are transcribed accurately.
"""

from __future__ import annotations

import logging
import signal
import threading

from karry_assistant.actions import apps as apps_actions
from karry_assistant.actions import media as media_actions
from karry_assistant.actions import open_things as open_actions
from karry_assistant.actions import power as power_actions
from karry_assistant.actions import volume as volume_actions
from karry_assistant.actions import web as web_actions
from karry_assistant.actions import youtube as youtube_actions
from karry_assistant.actions.registry import ActionRegistry, ActionResult, ToolSchema
from karry_assistant.audio.mic import MicStream
from karry_assistant.audio.vad import VadCommandRecorder
from karry_assistant.audio.wake import VoskWakeDetector
from karry_assistant.config import Settings, load_settings
from karry_assistant.nlp import intent as A
from karry_assistant.nlp.agent import Agent
from karry_assistant.nlp.intent import Intent
from karry_assistant.nlp.ollama_client import OllamaClient
from karry_assistant.nlp.rules import RulesParser
from karry_assistant.stt.whisper_stt import WhisperSTT
from karry_assistant.tts.speaker import Speaker
from karry_assistant.ui.events import KarryListener, NullListener
from karry_assistant.utils.logging import configure_logging
from karry_assistant.utils.text import classify_yes_no


logger = logging.getLogger("karry")


# ---------------------------------------------------------------------------
# Action wiring
# ---------------------------------------------------------------------------
def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


_EMPTY = _obj({})


def _build_registry() -> ActionRegistry:
    registry = ActionRegistry()

    # Power (destructive → require verbal confirmation)
    registry.register(
        A.ACTION_HIBERNATE,
        power_actions.hibernate,
        destructive=True,
        confirm_en="Should I hibernate the pc?",
        confirm_hi="PC hibernate karun?",
        schema=ToolSchema(
            description="Hibernate the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_SHUTDOWN,
        power_actions.shutdown,
        destructive=True,
        confirm_en="Should I shut down the pc?",
        confirm_hi="PC band karun?",
        schema=ToolSchema(
            description="Shut down the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_RESTART,
        power_actions.restart,
        destructive=True,
        confirm_en="Should I restart the pc?",
        confirm_hi="PC restart karun?",
        schema=ToolSchema(
            description="Restart the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_SLEEP,
        power_actions.sleep,
        schema=ToolSchema(description="Put the PC to sleep.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_LOCK,
        power_actions.lock,
        schema=ToolSchema(description="Lock the Windows session.", parameters=_EMPTY),
    )

    # Apps
    registry.register(
        A.ACTION_APP_LAUNCH,
        apps_actions.launch_app,
        schema=ToolSchema(
            description="Launch a Windows app by name (chrome, notepad, vs code, etc.).",
            parameters=_obj(
                {"app": {"type": "string", "description": "Application name."}},
                required=["app"],
            ),
        ),
    )

    # Volume
    registry.register(
        A.ACTION_VOLUME_UP,
        volume_actions.volume_up,
        schema=ToolSchema(description="Turn the master volume up one step.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_DOWN,
        volume_actions.volume_down,
        schema=ToolSchema(description="Turn the master volume down one step.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_MUTE,
        volume_actions.volume_mute,
        schema=ToolSchema(description="Mute the master volume.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_UNMUTE,
        volume_actions.volume_unmute,
        schema=ToolSchema(description="Unmute the master volume.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_SET,
        volume_actions.volume_set,
        schema=ToolSchema(
            description="Set master volume to a specific level (0-100).",
            parameters=_obj(
                {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                required=["level"],
            ),
        ),
    )

    # Media transport
    registry.register(
        A.ACTION_MEDIA_PLAY_PAUSE,
        media_actions.play_pause,
        schema=ToolSchema(description="Toggle play/pause on the active media player.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_NEXT,
        media_actions.next_track,
        schema=ToolSchema(description="Skip to the next track.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_PREV,
        media_actions.prev_track,
        schema=ToolSchema(description="Go to the previous track.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_STOP,
        media_actions.stop,
        schema=ToolSchema(description="Stop media playback.", parameters=_EMPTY),
    )

    # Open / Web / YouTube
    registry.register(
        A.ACTION_OPEN_THING,
        open_actions.open_thing,
        schema=ToolSchema(
            description="Open a URL, local file, or folder in its default handler.",
            parameters=_obj(
                {"target": {"type": "string", "description": "URL, absolute file path, or folder path."}},
                required=["target"],
            ),
        ),
    )
    registry.register(
        A.ACTION_WEB_SEARCH,
        web_actions.web_search,
        schema=ToolSchema(
            description="Open a Google search for the given query in the default browser.",
            parameters=_obj(
                {"query": {"type": "string", "description": "Search query."}},
                required=["query"],
            ),
        ),
    )
    registry.register(
        A.ACTION_YOUTUBE_PLAY,
        youtube_actions.play_on_youtube,
        schema=ToolSchema(
            description=(
                "Search YouTube for the given query and play the first result "
                "in the default browser. Keep the query in the user's original "
                "language (do NOT translate Hindi titles)."
            ),
            parameters=_obj(
                {
                    "query": {
                        "type": "string",
                        "description": "Song or video title, verbatim in the user's language.",
                    }
                },
                required=["query"],
            ),
        ),
    )

    return registry


# ---------------------------------------------------------------------------
# Karry — the running assistant object
# ---------------------------------------------------------------------------
class Karry:
    def __init__(
        self,
        settings: Settings,
        listener: KarryListener | None = None,
    ) -> None:
        self._settings = settings
        self._stop = threading.Event()
        self._listener: KarryListener = listener or NullListener()

        self._mic = MicStream(
            sample_rate=settings.sample_rate,
            device_index=settings.mic_device_index,
        )
        self._wake = VoskWakeDetector(
            model_path=str(settings.resolve_path(settings.vosk_model_path)),
            sample_rate=settings.sample_rate,
            wake_phrases=settings.wake_phrases,
            fuzzy_threshold=settings.wake_fuzzy_threshold,
        )
        self._vad = VadCommandRecorder(
            sample_rate=settings.sample_rate,
            aggressiveness=settings.vad_aggressiveness,
            silence_ms=settings.command_silence_ms,
            max_seconds=settings.command_max_seconds,
        )

        self._stt = WhisperSTT(
            model_name=settings.whisper_model,
            compute_type=settings.whisper_compute_type,
            device=settings.whisper_device,
            language=settings.whisper_language or None,
            download_root=str(settings.resolve_path(settings.whisper_model_dir)),
        )
        self._llm: OllamaClient | None = None
        if settings.ollama_enabled:
            self._llm = OllamaClient(
                url=settings.ollama_url,
                model=settings.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            )
        self._registry = _build_registry()
        self._rules = RulesParser()
        self._agent: Agent | None = None  # built lazily after mic is open
        self._speaker = Speaker(
            enabled=settings.tts_enabled,
            voice_en=settings.tts_voice_en,
            voice_hi=settings.tts_voice_hi,
            rate=settings.tts_rate,
        )

    # -- lifecycle ------------------------------------------------------
    def request_stop(self) -> None:
        logger.info("Stop requested")
        self._stop.set()

    def run(self) -> None:
        logger.info("Karry starting")
        self._listener.on_status("starting")
        # Open the mic FIRST so the Windows recording-in-use indicator
        # lights up immediately — that gives visible confirmation the app
        # is alive during long first-run downloads. Wake detection only
        # begins after Whisper is ready.
        with self._mic:
            print("[karry] microphone open (check the taskbar mic icon)")
            self._listener.on_status("microphone open")
            logger.info("Microphone opened; loading Whisper model...")
            print("[karry] loading Whisper model — first run may download ~1.6 GB, please wait...")
            self._listener.on_status("loading Whisper model...")
            try:
                self._stt._ensure_loaded()  # noqa: SLF001
                print("[karry] Whisper model ready")
                self._listener.on_status("Whisper ready")
            except Exception:  # noqa: BLE001
                logger.exception("Whisper preload failed; commands will not work")
                print("[karry] WARNING: Whisper failed to load — see log for details")
                self._listener.on_error("Whisper failed to load")

            # Verify Ollama at startup so the user sees a clear message
            # instead of silently degrading to rules-only.
            if self._llm is not None:
                try:
                    if self._llm.is_reachable():
                        print(
                            f"[karry] Ollama reachable at {self._settings.ollama_url} "
                            f"(model={self._settings.ollama_model})"
                        )
                        self._listener.on_status(
                            f"Ollama connected ({self._settings.ollama_model})"
                        )
                        # Build the agent once Ollama is confirmed.
                        self._agent = Agent(
                            registry=self._registry,
                            llm=self._llm,
                            confirm_cb=self._confirm_intent,
                            listener=self._listener,
                        )
                    else:
                        print(
                            f"[karry] WARNING: Ollama not reachable at "
                            f"{self._settings.ollama_url} — rules-only mode"
                        )
                        self._listener.on_error(
                            f"Ollama not reachable at {self._settings.ollama_url}"
                        )
                except Exception:  # noqa: BLE001
                    print("[karry] WARNING: could not probe Ollama; rules-only fallback active")
                    self._listener.on_error("could not probe Ollama; rules-only mode")

            self._mic.drain()  # discard audio buffered during model load
            print(f"[karry] listening. Say '{self._settings.wake_phrases[0]}'.")
            self._listener.on_status(f"listening for '{self._settings.wake_phrases[0]}'")
            self._speaker.speak("Karry is ready.", lang="en")

            while not self._stop.is_set():
                event = self._wake.wait_for_wake(self._mic, self._stop)
                if event is None:
                    break
                self._handle_wake(event.residual_text)
                self._listener.on_status(f"listening for '{self._settings.wake_phrases[0]}'")

        logger.info("Karry stopped")
        print("[karry] stopped")
        self._listener.on_status("stopped")
        try:
            self._speaker.stop()
        except Exception:  # noqa: BLE001
            pass

    # -- one turn -------------------------------------------------------
    def _handle_wake(self, residual: str) -> None:
        self._listener.on_status("wake detected")
        # Fast path: rules match on the residual (English/ASCII only).
        if residual:
            self._listener.on_heard(residual, "en")
            fast_intent = self._rules.parse(residual)
            if fast_intent is not None and not fast_intent.is_unknown:
                self._speaker.speak("Yes.", lang="en")
                self._execute_intent_directly(fast_intent, lang_hint="en")
                return

        # Prompt + capture command via VAD → Whisper → agent.
        self._speaker.speak_and_wait("Yes?", lang="en", timeout=4.0)
        self._listener.on_status("listening for command...")
        self._mic.drain()
        pcm = self._vad.record(self._mic, self._stop)
        if not pcm:
            logger.info("No command speech captured")
            self._listener.on_status("no speech captured")
            return

        self._listener.on_status("transcribing...")
        transcript = self._stt.transcribe_pcm(pcm, sample_rate=self._settings.sample_rate)
        if not transcript.text.strip():
            self._speaker.speak("I didn't catch that.", lang="en")
            self._listener.on_error("didn't catch that")
            return
        self._listener.on_heard(transcript.text, transcript.language)

        # Rules on the full transcript (still a fast-path — skips LLM).
        rule_intent = self._rules.parse(transcript.text)
        if rule_intent is not None and not rule_intent.is_unknown:
            print(f"[karry] rules matched: {rule_intent.name}")
            self._execute_intent_directly(rule_intent, lang_hint=transcript.language)
            return

        # Agentic path.
        if self._agent is None:
            msg = "AI backend not available"
            self._speaker.speak(f"Sorry, I need my AI backend running to handle that.", lang="en")
            self._listener.on_error(msg)
            return

        print(f"[karry] agent handling: {transcript.text!r}")
        self._listener.on_status("thinking...")
        outcome = self._agent.run(transcript.text)
        speech = self._agent.final_utterance(outcome, lang=transcript.language)
        if speech:
            self._listener.on_response(speech)
            self._speaker.speak(speech, lang=transcript.language)

    # -- direct dispatch (rules fast-path) ------------------------------
    def _execute_intent_directly(self, intent: Intent, lang_hint: str = "en") -> None:
        if intent.is_unknown:
            self._speaker.speak("Sorry, I didn't understand.", lang="en")
            self._listener.on_error("didn't understand")
            return
        if intent.is_cancel:
            self._speaker.speak("Okay, cancelled.", lang="en")
            self._listener.on_response("cancelled")
            return

        self._listener.on_tool_call(intent.name, dict(intent.slots))

        if (
            self._settings.confirm_destructive
            and self._registry.is_destructive(intent.name)
        ):
            if not self._confirm_intent(intent, lang_hint=lang_hint):
                self._speaker.speak("Okay, cancelled.", lang="en")
                self._listener.on_tool_result(intent.name, False, "user did not confirm")
                return

        result = self._registry.dispatch(intent)
        summary = result.speak_en if result.ok else result.error
        self._listener.on_tool_result(intent.name, result.ok, summary or "")
        self._speak_result(result, lang_hint)

    # -- confirmation callback used by BOTH direct-dispatch AND Agent --
    def _confirm_intent(self, intent: Intent, lang_hint: str = "en") -> bool:
        prompt = self._registry.confirm_prompt(intent.name, lang=lang_hint)
        logger.info("Confirmation prompt: %s", prompt)
        self._listener.on_confirm_prompt(intent.name, prompt)
        self._speaker.speak_and_wait(prompt, lang=lang_hint, timeout=6.0)

        self._mic.drain()
        pcm = self._vad.record(self._mic, self._stop)
        if not pcm:
            logger.info("No response — treating as no")
            self._listener.on_confirm_answer("(no answer)")
            return False
        answer = self._stt.transcribe_pcm(pcm, sample_rate=self._settings.sample_rate)
        verdict = classify_yes_no(answer.text)
        logger.info("Confirmation response %r → %s", answer.text, verdict)
        self._listener.on_confirm_answer(f"{answer.text!r} → {verdict}")
        return verdict == "yes"

    def _speak_result(self, result: ActionResult, lang_hint: str) -> None:
        if not self._settings.tts_enabled:
            return
        if result.ok:
            phrase = (
                result.speak_hi if (lang_hint == "hi" and result.speak_hi) else result.speak_en
            )
            if phrase:
                lang = "hi" if (lang_hint == "hi" and result.speak_hi) else "en"
                self._listener.on_response(phrase)
                self._speaker.speak(phrase, lang=lang)
        else:
            msg = f"Sorry, that didn't work: {result.error}"
            self._listener.on_error(result.error or "unknown error")
            self._speaker.speak(msg, lang="en")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    print("[karry] starting…  (Ctrl+C to quit)")
    print(f"[karry] whisper_model={settings.whisper_model}  ollama_model={settings.ollama_model}")
    logger.info("Loaded settings for Karry")

    karry = Karry(settings)

    def _signal_handler(signum: int, frame: object) -> None:
        del signum, frame
        karry.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (AttributeError, ValueError):
        pass

    karry.run()


if __name__ == "__main__":
    run()
