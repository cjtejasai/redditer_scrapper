## Luxury Lead Watcher (FastAPI)

This runs your notebook workflow as a small FastAPI app with a UI that shows the final leads.

### 1) Create venv + install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.fastapi.txt
```

### 2) Configure keys

```powershell
Copy-Item .env.example .env
notepad .env
```

Required: `FIRECRAWL_API_KEY`  
Recommended: `APIFY_TOKEN` + `OPENAI_API_KEY` (better classification)

### 3) Run

```powershell
uvicorn luxwatcher.main:app --port 8000
```

Open `http://127.0.0.1:8000`

### Notes
- Don’t use `--reload` here: writing `data/*.csv` triggers auto-reloads and interrupts the scrape.
- The app runs once on startup, then every `SCRAPE_INTERVAL_HOURS` (default `24`).
- Outputs are written to `data/leads_only.csv`, `data/all_results.csv`, and `data/firecrawl_hits_YYYYMMDD.csv`.
- Firecrawl runs every time by default; set `USE_HITS_CACHE=1` to reuse the same-day hits cache.
- Previous runs are snapshotted under `data/runs/<run_id>/` and selectable in the UI dropdown.
- If it feels “stuck” after Firecrawl, it’s usually Apify taking time; you can reduce work with `APIFY_MAX_URLS` and `MAX_URLS` in `.env`.
- If you get `ApifyApiError: Insufficient permissions for the Actor`, either set `APIFY_MAX_URLS=0` (skip Apify) or set `APIFY_ACTOR_ID` to an actor your token can run.
