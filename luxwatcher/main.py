from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi import Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from luxwatcher.pipeline import load_config, run_once
from luxwatcher.storage import read_csv_dicts
from luxwatcher.notifications import send_leads_notification


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)


class RuntimeState:
    def __init__(self) -> None:
        self.run_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.running = False
        self.current_run_id: str | None = None
        self.last_started: str | None = None
        self.last_finished: str | None = None
        self.last_error: str | None = None
        self.last_summary: dict | None = None
        self.stage: str | None = None
        self.stage_detail: str | None = None
        self.run_history: list[dict] = []
        self.shutdown = asyncio.Event()


state = RuntimeState()

def _key_fingerprint(value: str | None) -> str | None:
    v = (value or "").strip()
    if not v:
        return None
    return hashlib.sha256(v.encode("utf-8", "ignore")).hexdigest()[:10]


def _is_excluded_url(url: str) -> bool:
    u = (url or "").lower()
    return "reddit.com/r/digitalnomad" in u


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")


def _runs_dir(data_dir: Path) -> Path:
    return data_dir / "runs"


def _persist_run_meta(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_run_history_from_disk(data_dir: Path, limit: int = 50) -> list[dict]:
    runs_dir = _runs_dir(data_dir)
    if not runs_dir.exists():
        return []

    items: list[dict] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
            if isinstance(meta, dict):
                items.append(meta)
                continue

        leads_csv = d / "leads_only.csv"
        if leads_csv.exists():
            items.append(
                {
                    "run_id": d.name,
                    "started_utc": None,
                    "finished_utc": None,
                    "error": None,
                    "summary": None,
                    "outputs": {"leads_out": str(leads_csv)},
                }
            )

    def _sort_key(m: dict) -> str:
        return (m.get("finished_utc") or m.get("started_utc") or m.get("run_id") or "")

    items.sort(key=_sort_key, reverse=True)
    return items[:limit]


def _snapshot_outputs(run_dir: Path, summary: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for k in ["hits_csv", "all_out", "leads_out"]:
        v = summary.get(k)
        if not v:
            continue
        src = Path(str(v))
        if not src.exists():
            continue
        dest = run_dir / src.name
        try:
            shutil.copyfile(src, dest)
            out[k] = str(dest)
        except Exception:
            continue
    return out


def _set_status(*, stage: str | None = None, stage_detail: str | None = None) -> None:
    with state.status_lock:
        if stage is not None:
            state.stage = stage
        if stage_detail is not None:
            state.stage_detail = stage_detail


async def _run_pipeline_background() -> None:
    if not state.run_lock.acquire(blocking=False):
        return
    try:
        with state.status_lock:
            state.running = True
            state.last_error = None
            state.current_run_id = _new_run_id()
            state.last_started = datetime.now(timezone.utc).isoformat()
            state.last_finished = None
            state.last_summary = None
        _set_status(stage="starting", stage_detail="Starting pipeline…")
        try:
            cfg = load_config()
            summary = await asyncio.to_thread(run_once, cfg, _set_status)
            run_id = state.current_run_id or _new_run_id()
            run_dir = _runs_dir(cfg.data_dir) / run_id
            outputs = _snapshot_outputs(run_dir, summary)
            meta = {
                "run_id": run_id,
                "started_utc": state.last_started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "error": None,
                "summary": dict(summary),
                "outputs": outputs,
            }
            _persist_run_meta(run_dir, meta)
            with state.status_lock:
                state.last_summary = summary
                state.last_finished = datetime.now(timezone.utc).isoformat()
                state.run_history.insert(0, meta)
                try:
                    limit = int(os.getenv("RUN_HISTORY_LIMIT", "50"))
                    state.run_history = state.run_history[:limit]
                except Exception:
                    pass
            _set_status(stage="done", stage_detail=f"Done. Leads: {summary.get('leads')} / Total: {summary.get('total')}")

            # Send email notification on success
            if os.getenv("SENDGRID_API_KEY"):
                try:
                    leads_data = []
                    leads_csv = summary.get("leads_out")
                    if leads_csv:
                        leads_data = read_csv_dicts(Path(leads_csv), limit=10)

                    dashboard_url = os.getenv("DASHBOARD_URL") or os.getenv("NGROK_DOMAIN")
                    if dashboard_url and not dashboard_url.startswith("http"):
                        dashboard_url = f"https://{dashboard_url}"

                    notif_result = send_leads_notification(
                        leads_count=summary.get("leads", 0),
                        total_count=summary.get("total", 0),
                        run_id=run_id,
                        leads_data=leads_data,
                        dashboard_url=dashboard_url,
                    )
                    if notif_result.get("success"):
                        _set_status(stage="done", stage_detail=f"Done. Leads: {summary.get('leads')} / Total: {summary.get('total')} | Email sent")
                except Exception as notif_err:
                    pass  # Don't fail the run if notification fails
        except Exception as e:
            try:
                cfg = load_config()
                run_id = state.current_run_id or _new_run_id()
                run_dir = _runs_dir(cfg.data_dir) / run_id
                meta = {
                    "run_id": run_id,
                    "started_utc": state.last_started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(e).__name__}: {e}",
                    "summary": None,
                    "outputs": {},
                }
                _persist_run_meta(run_dir, meta)
                with state.status_lock:
                    state.run_history.insert(0, meta)
                    try:
                        limit = int(os.getenv("RUN_HISTORY_LIMIT", "50"))
                        state.run_history = state.run_history[:limit]
                    except Exception:
                        pass
            except Exception:
                pass
            with state.status_lock:
                state.last_error = f"{type(e).__name__}: {e}"
                state.last_finished = datetime.now(timezone.utc).isoformat()
            _set_status(stage="error", stage_detail=state.last_error)
        finally:
            with state.status_lock:
                state.running = False
                state.current_run_id = None
    finally:
        state.run_lock.release()


async def _periodic_runner(interval_hours: float) -> None:
    await _run_pipeline_background()
    while not state.shutdown.is_set():
        try:
            await asyncio.wait_for(state.shutdown.wait(), timeout=interval_hours * 3600)
        except asyncio.TimeoutError:
            await _run_pipeline_background()


def _send_daily_email() -> None:
    """Send daily scheduled email with latest leads."""
    if not os.getenv("SENDGRID_API_KEY"):
        return

    try:
        cfg = load_config()
        leads_csv = cfg.data_dir / "leads_only.csv"

        if not leads_csv.exists():
            return

        leads_data = read_csv_dicts(leads_csv, limit=50)
        leads_count = len(leads_data)

        if leads_count == 0:
            return

        dashboard_url = os.getenv("DASHBOARD_URL") or os.getenv("NGROK_DOMAIN")
        if dashboard_url and not dashboard_url.startswith("http"):
            dashboard_url = f"https://{dashboard_url}"

        send_leads_notification(
            leads_count=leads_count,
            total_count=leads_count,
            run_id=f"daily_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            leads_data=leads_data,
            dashboard_url=dashboard_url,
        )
    except Exception:
        pass  # Don't crash on email failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        cfg = load_config()
        limit = int(os.getenv("RUN_HISTORY_LIMIT", "50"))
        history = _load_run_history_from_disk(cfg.data_dir, limit=limit)
        with state.status_lock:
            state.run_history = history
    except Exception:
        pass

    # Start pipeline runner
    interval_hours = float(os.getenv("SCRAPE_INTERVAL_HOURS", "24"))
    task = asyncio.create_task(_periodic_runner(interval_hours))

    # Email is sent after pipeline completes (in _run_pipeline_background)
    # No separate scheduler needed

    try:
        yield
    finally:
        state.shutdown.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


app = FastAPI(title="Luxury Lead Watcher", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (templates_dir / "index.html").read_text(encoding="utf-8")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/status")
def api_status():
    cfg = load_config()
    leads_csv = cfg.data_dir / "leads_only.csv"
    with state.status_lock:
        return {
            "running": state.running,
            "current_run_id": state.current_run_id,
            "last_started": state.last_started,
            "last_finished": state.last_finished,
            "last_error": state.last_error,
            "last_summary": state.last_summary,
            "stage": state.stage,
            "stage_detail": state.stage_detail,
            "leads_path": str(leads_csv),
            "runs_count": len(state.run_history),
            "latest_run_id": (state.run_history[0].get("run_id") if state.run_history else None),
            "classifier": "perplexity"
            if bool(cfg.perplexity_api_key)
            else ("openai" if bool(cfg.openai_api_key) else ("gemini" if bool(cfg.gemini_api_key) else "heuristic")),
            "perplexity_model": cfg.perplexity_model if bool(cfg.perplexity_api_key) else None,
            "perplexity_key_fp": _key_fingerprint(cfg.perplexity_api_key),
            "openai_model": cfg.openai_model if bool(cfg.openai_api_key) else None,
            "openai_key_fp": _key_fingerprint(cfg.openai_api_key),
        }


@app.post("/api/run")
async def api_run(background: BackgroundTasks):
    with state.status_lock:
        if state.running:
            raise HTTPException(status_code=409, detail="Run already in progress")
    background.add_task(_run_pipeline_background)
    return {"ok": True}


@app.get("/api/leads")
def api_leads(
    limit: int = Query(200, ge=1, le=2000),
    run_id: str | None = None,
    include_all: bool = Query(False, description="Include rejected leads (all results)")
):
    cfg = load_config()
    # Choose file based on include_all flag
    if include_all:
        csv_file = cfg.data_dir / "all_results.csv"
    else:
        csv_file = cfg.data_dir / "leads_only.csv"

    if run_id:
        with state.status_lock:
            meta = next((m for m in state.run_history if m.get("run_id") == run_id), None)
        if meta:
            if include_all:
                p = (meta.get("outputs") or {}).get("all_out") or (cfg.data_dir / "runs" / run_id / "all_results.csv")
            else:
                p = (meta.get("outputs") or {}).get("leads_out") or (cfg.data_dir / "runs" / run_id / "leads_only.csv")
            csv_file = Path(str(p))

    def _confidence(it: dict) -> float:
        try:
            return float(it.get("confidence") or 0.0)
        except Exception:
            return 0.0

    items = read_csv_dicts(csv_file, limit=None)
    items = [it for it in items if not _is_excluded_url(it.get("url") or "")]
    items.sort(key=_confidence, reverse=True)
    items = items[:limit]
    return {"items": items, "run_id": run_id, "include_all": include_all}


@app.get("/api/runs")
def api_runs(limit: int = Query(50, ge=1, le=200)):
    cfg = load_config()
    items = _load_run_history_from_disk(cfg.data_dir, limit=limit)
    with state.status_lock:
        state.run_history = items
        latest_run_id = items[0].get("run_id") if items else None
    return {"items": items, "latest_run_id": latest_run_id}
