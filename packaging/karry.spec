# PyInstaller spec for Karry.
# Build with:  pyinstaller packaging/karry.spec
#
# Output:  dist/karry/karry.exe  (onedir layout — recommended for large ML deps)
#
# Notes:
# - Whisper models are NOT bundled. On first run, faster-whisper downloads
#   the configured model into `models/whisper/` next to the exe (see
#   Settings.whisper_model_dir). This keeps the exe small (~50-80 MB)
#   and lets the user swap models without a rebuild.
# - Vosk small-en model must be extracted into `models/vosk-model-small-en-us-0.15/`
#   next to the exe before first run (~50 MB one-time download).
# - Ollama is a separate installation. The app degrades gracefully to
#   rules-only if Ollama is not reachable.

from __future__ import annotations

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(SPECPATH), os.pardir)) if False else os.path.abspath(os.path.dirname(SPECPATH) + os.sep + os.pardir)
# ^^^ SPECPATH is provided by PyInstaller; the above evaluates to the repo root.


hiddenimports = []
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("vosk")
hiddenimports += collect_submodules("sounddevice")
hiddenimports += collect_submodules("edge_tts")
hiddenimports += collect_submodules("pyttsx3")
hiddenimports += collect_submodules("pystray")
hiddenimports += collect_submodules("PIL")
hiddenimports += collect_submodules("comtypes")
hiddenimports += collect_submodules("pycaw")
hiddenimports += collect_submodules("pywhatkit")


datas = []
datas += collect_data_files("faster_whisper", include_py_files=False)
datas += collect_data_files("ctranslate2", include_py_files=False)
datas += collect_data_files("vosk", include_py_files=False)
datas += collect_data_files("sounddevice", include_py_files=False)


block_cipher = None


a = Analysis(
    [os.path.join(REPO_ROOT, "run.py")],
    pathex=[os.path.join(REPO_ROOT, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "scipy",
        "pandas",
        "notebook",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="karry",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,             # windowed app; logs go to %APPDATA%\Karry\logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="karry",
)
