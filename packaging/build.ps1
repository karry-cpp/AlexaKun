# Karry build helper (PowerShell). Run from the repo root:
#   .\packaging\build.ps1
#
# Prereqs: activate your venv (or ensure pip + pyinstaller are in PATH).

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "== Cleaning previous build/dist ==" -ForegroundColor Cyan
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist

Write-Host "== Ensuring PyInstaller is installed ==" -ForegroundColor Cyan
python -m pip install --upgrade pyinstaller

Write-Host "== Building Karry ==" -ForegroundColor Cyan
python -m PyInstaller "packaging/karry.spec" --noconfirm

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "  Executable: $repoRoot\dist\karry\karry.exe"
Write-Host ""
Write-Host "First-run checklist:" -ForegroundColor Yellow
Write-Host "  1. Extract vosk-model-small-en-us-0.15 into dist\karry\models\"
Write-Host "  2. Install Ollama and run: ollama pull qwen2.5:3b-instruct"
Write-Host "  3. Copy .env.example to dist\karry\.env if you want to tweak defaults"
Write-Host "  4. Launch: dist\karry\karry.exe"
