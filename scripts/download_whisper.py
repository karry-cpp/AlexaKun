"""Pre-download a faster-whisper model with a visible progress bar.

Usage (from repo root, with the venv activated):

    python scripts/download_whisper.py                     # uses KARRY_WHISPER_MODEL from .env / defaults
    python scripts/download_whisper.py --model small        # override
    python scripts/download_whisper.py --model large-v3-turbo
    python scripts/download_whisper.py --hf-token hf_xxx    # avoid anonymous rate limiting

Why this script exists
----------------------
When faster-whisper downloads a model on its very first `WhisperModel(...)`
call inside Karry, HF Hub's programmatic client sometimes shows no
visible progress and anonymous downloads can be heavily throttled.
Running this script pre-populates the cache at ``models/whisper/`` so
that Karry's first run loads the model from disk in seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the src/ package importable so we can read the same defaults as Karry.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Silence the harmless Windows symlink warning before importing hf.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --- HuggingFace repo mapping ------------------------------------------------
# faster-whisper looks up model names against these HuggingFace repos.
# Same table as inside faster-whisper's utils.py.
_HF_REPOS = {
    "tiny":             "Systran/faster-whisper-tiny",
    "tiny.en":          "Systran/faster-whisper-tiny.en",
    "base":             "Systran/faster-whisper-base",
    "base.en":          "Systran/faster-whisper-base.en",
    "small":            "Systran/faster-whisper-small",
    "small.en":         "Systran/faster-whisper-small.en",
    "medium":           "Systran/faster-whisper-medium",
    "medium.en":        "Systran/faster-whisper-medium.en",
    "large-v1":         "Systran/faster-whisper-large-v1",
    "large-v2":         "Systran/faster-whisper-large-v2",
    "large-v3":         "Systran/faster-whisper-large-v3",
    "large":            "Systran/faster-whisper-large-v3",
    "distil-large-v2":  "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3":  "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo":   "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo":            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _resolve_repo(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in _HF_REPOS:
        # Assume the user passed a full HF repo id like "org/name".
        if "/" in model_name:
            return model_name
        raise SystemExit(
            f"Unknown model {model_name!r}. Known: {', '.join(sorted(_HF_REPOS))}"
        )
    return _HF_REPOS[key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download a Whisper model for Karry.")
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper model name (tiny/base/small/medium/large-v3/large-v3-turbo). "
        "Defaults to KARRY_WHISPER_MODEL from settings.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory. Defaults to KARRY_WHISPER_MODEL_DIR from settings.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token to avoid anonymous rate limiting.",
    )
    args = parser.parse_args()

    # Pull defaults from the same settings Karry uses.
    from karry_assistant.config import load_settings

    settings = load_settings()
    model_name = args.model or settings.whisper_model
    cache_dir = args.cache_dir or str(settings.resolve_path(settings.whisper_model_dir))
    repo_id = _resolve_repo(model_name)

    print(f"[download] model  : {model_name}")
    print(f"[download] repo   : {repo_id}")
    print(f"[download] target : {cache_dir}")
    print(f"[download] token  : {'set' if args.hf_token else 'anonymous (may be slow — see README)'}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # snapshot_download shows a tqdm progress bar per file when called from
    # a real terminal — much clearer than the silent behaviour inside
    # WhisperModel().
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=args.hf_token,
            allow_patterns=["*.bin", "*.json", "*.txt", "tokenizer*", "vocab*"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[download] FAILED: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print(f"[download] done in {elapsed:.0f}s -> {cache_dir}")
    print("[download] you can now run: python run.py --console")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
