$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

if (!(Test-Path ".\\.venv")) {
  python -m venv .venv
}

.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.fastapi.txt --upgrade

if (!(Test-Path ".\\.env")) {
  Copy-Item .env.example .env
  Write-Host "Created .env from .env.example. Please set FIRECRAWL_API_KEY (and optional APIFY/GEMINI keys)."
}

uvicorn luxwatcher.main:app --port 8000
