from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from luxwatcher.queries import get_queries
from luxwatcher.storage import append_rows, ensure_csv_has_header, ensure_dir


FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_dir: Path = REPO_ROOT / "data"
    tbs: str = "qdr:d"  # last 24 hours
    limit: int = 10
    firecrawl_api_key: str | None = None
    apify_token: str | None = None
    apify_actor_id: str = "oAuCIx3ItNrs2okjQ"
    apify_max_urls: int = 60
    max_urls: int = 300
    use_hits_cache: bool = False
    hits_max_age_hours: float = 24.0
    perplexity_api_key: str | None = None
    perplexity_model: str = "sonar"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_organization: str | None = None
    openai_project: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    batch_size: int = 20
    max_comments_per_post: int = 30
    max_total_chars: int = 14000
    sleep_between_llm_calls_sec: float = 1.0


def load_config() -> Config:
    try:
        from dotenv import load_dotenv  # optional when running outside FastAPI

        load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
    except Exception:
        pass

    def _env_bool(name: str, default: str = "1") -> bool:
        v = (os.getenv(name, default) or "").strip().lower()
        return v in {"1", "true", "yes", "y", "on"}

    data_dir_env = (os.getenv("DATA_DIR") or "data").strip()
    data_dir = Path(data_dir_env).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    return Config(
        data_dir=data_dir,
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        apify_token=os.getenv("APIFY_TOKEN"),
        apify_actor_id=os.getenv("APIFY_ACTOR_ID", "oAuCIx3ItNrs2okjQ"),
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        perplexity_model=os.getenv("PERPLEXITY_MODEL", "sonar"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_organization=os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION"),
        openai_project=os.getenv("OPENAI_PROJECT"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        tbs=os.getenv("SCRAPE_TBS", "qdr:d"),
        limit=int(os.getenv("FIRECRAWL_LIMIT", "10")),
        apify_max_urls=int(os.getenv("APIFY_MAX_URLS", "60")),
        max_urls=int(os.getenv("MAX_URLS", "300")),
        use_hits_cache=_env_bool("USE_HITS_CACHE", "0"),
        hits_max_age_hours=float(os.getenv("HITS_MAX_AGE_HOURS", "24")),
        batch_size=int(os.getenv("BATCH_SIZE", "20")),
        sleep_between_llm_calls_sec=float(
            os.getenv("SLEEP_BETWEEN_LLM_CALLS_SEC") or os.getenv("SLEEP_BETWEEN_GEMINI_CALLS_SEC") or "1.0"
        ),
    )


def _stable_id_from_hit(url: str, title: str, snippet: str) -> str:
    raw = (url + title + snippet).encode("utf-8", "ignore")
    return hashlib.sha1(raw).hexdigest()


def _extract_firecrawl_results(resp_json: dict) -> list[dict]:
    data = resp_json.get("data", {})
    if isinstance(data, dict):
        web_results = data.get("web")
        if isinstance(web_results, list):
            return web_results
    return []


def _post_with_backoff(url: str, headers: dict, payload: dict, max_retries: int = 8, timeout: int = 60) -> requests.Response:
    for attempt in range(max_retries):
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait: float | None = None
            if retry_after:
                try:
                    wait = float(retry_after)
                except Exception:
                    wait = None
            if wait is None:
                wait = min(60.0, (2.0**attempt)) + random.uniform(0.6, 1.6)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError("Exceeded retries due to repeated 429 rate limits.")


def firecrawl_collect_hits(cfg: Config, queries: list[str]) -> Path:
    if not cfg.firecrawl_api_key:
        raise RuntimeError("Missing FIRECRAWL_API_KEY in environment.")

    ensure_dir(cfg.data_dir)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    hits_csv = cfg.data_dir / f"firecrawl_hits_{day}.csv"

    fieldnames = ["ts_utc", "query", "title", "url", "snippet", "position", "stable_id"]
    ensure_csv_has_header(hits_csv, fieldnames)

    # load seen IDs (avoid duplicates on rerun)
    seen: set[str] = set()
    if hits_csv.exists() and hits_csv.stat().st_size > 0:
        with hits_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("stable_id") or "").strip()
                if sid:
                    seen.add(sid)

    headers = {"Authorization": f"Bearer {cfg.firecrawl_api_key}", "Content-Type": "application/json"}
    base_payload = {
        "sources": ["web"],
        "categories": [],
        "tbs": cfg.tbs,
        "limit": cfg.limit,
        "scrapeOptions": {"onlyMainContent": False, "maxAge": 172800000, "parsers": ["pdf"], "formats": []},
    }

    new_rows: list[dict] = []
    for q in queries:
        payload = dict(base_payload)
        payload["query"] = q
        time.sleep(random.uniform(0.8, 1.8))
        resp = _post_with_backoff(FIRECRAWL_SEARCH_URL, headers, payload).json()
        results = _extract_firecrawl_results(resp)
        for item in results:
            url = item.get("url") or ""
            title = item.get("title") or ""
            snippet = item.get("description") or ""
            sid = _stable_id_from_hit(url, title, snippet)
            if sid in seen:
                continue
            seen.add(sid)
            new_rows.append(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "query": q,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "position": item.get("position"),
                    "stable_id": sid,
                }
            )

    append_rows(hits_csv, fieldnames, new_rows)
    return hits_csv


def firecrawl_scrape_url(cfg: Config, url: str, status_cb: Callable[..., None] | None = None) -> str | None:
    """
    Scrape a single URL using Firecrawl's /scrape endpoint.
    Returns the full markdown content of the page, or None on failure.
    """
    if not cfg.firecrawl_api_key:
        return None

    headers = {"Authorization": f"Bearer {cfg.firecrawl_api_key}", "Content-Type": "application/json"}
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": 2000,  # Wait 2s for JS to render
    }

    try:
        resp = _post_with_backoff(FIRECRAWL_SCRAPE_URL, headers, payload, max_retries=3, timeout=90)
        data = resp.json()

        # Extract markdown from response
        if data.get("success"):
            content_data = data.get("data", {})
            markdown = content_data.get("markdown", "")
            if markdown:
                return markdown

        return None
    except Exception as e:
        if status_cb:
            status_cb(stage="scrape", stage_detail=f"Scrape error for {url}: {type(e).__name__}")
        return None


def firecrawl_scrape_batch(
    cfg: Config,
    urls: list[str],
    status_cb: Callable[..., None] | None = None
) -> dict[str, str]:
    """
    Scrape multiple URLs using Firecrawl.
    Returns a dict of {url: markdown_content}.
    """
    results: dict[str, str] = {}
    total = len(urls)

    for i, url in enumerate(urls):
        if status_cb and (i % 5 == 0 or i == total - 1):
            status_cb(stage="scrape", stage_detail=f"Scraping {i+1}/{total} URLs with Firecrawl")

        content = firecrawl_scrape_url(cfg, url, status_cb)
        if content:
            results[url] = content

        # Rate limiting - be gentle with the API
        time.sleep(random.uniform(0.5, 1.0))

    return results


def _hits_csv_path(cfg: Config) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return cfg.data_dir / f"firecrawl_hits_{day}.csv"


def _is_recent_file(path: Path, max_age_hours: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    age_sec = max(0.0, time.time() - path.stat().st_mtime)
    return age_sec <= max_age_hours * 3600.0


def _normalize_reddit_url(url: str) -> str:
    url = (url or "").strip()
    url = url.split("?")[0]
    url = url.replace("old.reddit.com", "www.reddit.com")
    return url


_EXCLUDED_REDDIT_PATH_PREFIXES = [
    "/r/digitalnomad",
]

def _is_allowed_subreddit(url: str) -> bool:
    """Check if URL is from a valid subreddit (no filtering)."""
    u = (url or "").lower()
    if "reddit.com" not in u:
        return False
    # Just check it has a subreddit pattern
    return bool(re.search(r"/r/[^/]+", u))


def _is_excluded_reddit_url(url: str) -> bool:
    u = (url or "").lower()
    if "reddit.com" not in u:
        return False
    path = u.split("reddit.com", 1)[1]
    return any(prefix in path for prefix in _EXCLUDED_REDDIT_PATH_PREFIXES)


def _is_reddit_post_or_comment_url(url: str) -> bool:
    u = (url or "").lower()
    if "reddit.com" not in u:
        return False
    # Permalinks to posts/comments contain /comments/<post_id>/...
    return "/comments/" in u


def _read_unique_urls_from_hits_csv(path: Path) -> list[str]:
    urls: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = _normalize_reddit_url(row.get("url", ""))
            if u:
                urls.append(u)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _chunks(lst: list[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _run_apify_batch(cfg: Config, urls: list[str], status_cb: Callable[..., None] | None = None) -> list[dict]:
    if not cfg.apify_token:
        return []
    try:
        from apify_client import ApifyClient
    except Exception:
        return []

    apify = ApifyClient(cfg.apify_token)
    run_input = {
        "startUrls": [{"url": u} for u in urls],
        "skipComments": False,
        "skipUserPosts": False,
        "skipCommunity": False,
        "ignoreStartUrls": False,
        "searchPosts": False,
        "searchComments": False,
        "searchCommunities": False,
        "searchUsers": False,
        "sort": "new",
        "includeNSFW": True,
        "maxItems": 10000,
        "maxPostCount": len(urls),
        "maxComments": cfg.max_comments_per_post,
        "scrollTimeout": 40,
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
    }
    try:
        run = apify.actor(cfg.apify_actor_id).call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]
        return list(apify.dataset(dataset_id).iterate_items())
    except Exception as e:
        if status_cb:
            status_cb(stage="apify", stage_detail=f"Apify error: {type(e).__name__}: {e}")
        return []


def _group_items(items: list[dict]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    from collections import defaultdict

    posts: dict[str, dict] = {}
    comments_by_post: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it.get("dataType") == "post":
            pid = it.get("id")
            if pid:
                posts[str(pid)] = it
        elif it.get("dataType") == "comment":
            pid = it.get("postId")
            if pid:
                comments_by_post[str(pid)].append(it)
    return posts, comments_by_post


def _build_thread_text(cfg: Config, post: dict, comments: list[dict]) -> str:
    url = post.get("url") or post.get("link") or ""
    title = post.get("title", "")
    body = post.get("body", "")

    comments_sorted = sorted(comments, key=lambda c: c.get("upVotes") or 0, reverse=True)[: cfg.max_comments_per_post]
    parts = [f"URL: {url}", f"TITLE: {title}", "POST:", body, "", "COMMENTS:"]
    for i, c in enumerate(comments_sorted, 1):
        parts.append(f"{i}. @{c.get('username','')}: {c.get('body','')}")
        parts.append(f"   ({c.get('url','')})")
    return _truncate("\n".join(parts), cfg.max_total_chars)


def _extract_json_loose(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _gemini_classify(cfg: Config, thread_text: str) -> dict:
    if not cfg.gemini_api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")
    from google import genai

    gemini = genai.Client(api_key=cfg.gemini_api_key)
    prompt = f"""
You are a strict lead qualifier for real estate buyer intent on Reddit.

Analyze the thread to identify if the original poster (OP) or any commenter is actively looking to BUY property.

Hard rules:
- EXCLUDE renters: People looking to RENT, LEASE, or CHARTER are NOT leads.
- Only qualify as lead if someone expresses clear intent to PURCHASE/BUY property.
- Focus on real estate only (apartments, houses, villas, property).

For each potential lead found, determine:
- source: "op" if original poster, "commenter" if from comments
- buyer_stage: "researching" (early research), "actively_looking" (actively searching), "ready_to_buy" (imminent purchase)
- property_type: "apartment", "house", "villa", "townhome", "residential", "holiday house or apartment", "unknown"
- budget: Extract any mentioned budget/price or "not_mentioned"
- thread_type: "buyer_seeking" (OP wants to buy), "general_discussion" (discussion where buyer emerged)

Return ONLY JSON with these keys:
{{
  "is_lead": true/false,
  "source": "op" | "commenter",
  "buyer_stage": "researching" | "actively_looking" | "ready_to_buy",
  "confidence": number between 0 and 1,
  "property_type": "apartment" | "house" | "villa" | "townhome" | "residential" | "holiday house or apartment" | "unknown",
  "budget": "extracted budget string or not_mentioned",
  "evidence": "short direct quote proving buying intent",
  "reason": "explanation of why this is a valuable lead for sales team",
  "thread_type": "buyer_seeking" | "general_discussion",
  "thread_summary": "one sentence summary of the thread"
}}

THREAD:
\"\"\"{thread_text}\"\"\"
""".strip()

    resp = gemini.models.generate_content(model=cfg.gemini_model, contents=prompt)
    data = _extract_json_loose(getattr(resp, "text", "") or "")
    data.setdefault("is_lead", False)
    data.setdefault("source", "op")
    data.setdefault("buyer_stage", "researching")
    data.setdefault("confidence", 0.0)
    data.setdefault("property_type", "unknown")
    data.setdefault("budget", "not_mentioned")
    data.setdefault("evidence", "")
    data.setdefault("reason", "")
    data.setdefault("thread_type", "general_discussion")
    data.setdefault("thread_summary", "")
    return data


def _openai_classify(cfg: Config, thread_text: str) -> dict:
    if not cfg.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    from openai import OpenAI

    timeout_sec = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    client = OpenAI(
        api_key=cfg.openai_api_key,
        organization=cfg.openai_organization,
        project=cfg.openai_project,
        timeout=timeout_sec,
        max_retries=2,
    )
    prompt = f"""
You are a strict lead qualifier for real estate buyer intent on Reddit.

Analyze the thread to identify if the original poster (OP) or any commenter is actively looking to BUY property.

Hard rules:
- EXCLUDE renters: People looking to RENT, LEASE, or CHARTER are NOT leads.
- Only qualify as lead if someone expresses clear intent to PURCHASE/BUY property.
- Focus on real estate only (apartments, houses, villas, property).

For each potential lead found, determine:
- source: "op" if original poster, "commenter" if from comments
- buyer_stage: "researching" (early research), "actively_looking" (actively searching), "ready_to_buy" (imminent purchase)
- property_type: "apartment", "house", "villa", "townhome", "residential", "holiday house or apartment", "unknown"
- budget: Extract any mentioned budget/price or "not_mentioned"
- thread_type: "buyer_seeking" (OP wants to buy), "general_discussion" (discussion where buyer emerged)

Return ONLY JSON with these keys:
{{
  "is_lead": true/false,
  "source": "op" | "commenter",
  "buyer_stage": "researching" | "actively_looking" | "ready_to_buy",
  "confidence": number between 0 and 1,
  "property_type": "apartment" | "house" | "villa" | "townhome" | "residential" | "holiday house or apartment" | "unknown",
  "budget": "extracted budget string or not_mentioned",
  "evidence": "short direct quote proving buying intent",
  "reason": "explanation of why this is a valuable lead for sales team",
  "thread_type": "buyer_seeking" | "general_discussion",
  "thread_summary": "one sentence summary of the thread"
}}

THREAD:
\"\"\"{thread_text}\"\"\"
""".strip()

    try:
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON (no markdown, no commentary)."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON (no markdown, no commentary)."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

    text = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""

    data = _extract_json_loose(text)
    data.setdefault("is_lead", False)
    data.setdefault("source", "op")
    data.setdefault("buyer_stage", "researching")
    data.setdefault("confidence", 0.0)
    data.setdefault("property_type", "unknown")
    data.setdefault("budget", "not_mentioned")
    data.setdefault("evidence", "")
    data.setdefault("reason", "")
    data.setdefault("thread_type", "general_discussion")
    data.setdefault("thread_summary", "")
    return data


def _perplexity_classify(cfg: Config, url: str, title: str, snippet: str, query: str) -> dict:
    if not cfg.perplexity_api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY.")

    master_prompt = (os.getenv("PERPLEXITY_MASTER_PROMPT") or "").strip()
    if not master_prompt:
        master_prompt = """
You are a strict lead qualifier for luxury intent on Reddit.

Visit and read the FULL content of the Reddit URL provided (post body and comments). Based on the actual content, decide if it contains a VALID actionable lead in any of:
- real_estate (buyer looking to BUY, seller/owner selling, or broker/agent)
- luxury_car (buyer looking to BUY, seller selling, or dealer)
- luxury_watch (buyer looking to BUY, seller selling, or dealer/AD)
- yacht (buyer looking to BUY, seller selling, or broker)
- private_jet (buyer looking to BUY, seller selling, or broker)

Hard rules:
- You MUST visit the URL and read the full post content and comments before classifying.
- EXCLUDE renters: People looking to RENT, LEASE, or CHARTER are NOT leads. Return not_lead for them.
- Only classify as lead if: someone wants to BUY, someone is SELLING/owns and selling, or someone is a BROKER/AGENT/DEALER.
- Sellers and owners selling should be classified as lead_type "broker".
- If the content is NOT about these verticals (e.g. workplace drama, legal/HR issues, tech support, relationships, health, etc.), return not_lead.
- Do NOT infer a vertical from vibes. Only classify a lead when there is explicit intent to buy or sell.
- Evidence must be a short direct quote from the post content (not invented). If you cannot quote intent, return not_lead.

Return ONLY JSON with these keys:
{
  "is_lead": true/false,
  "vertical": "real_estate" | "luxury_car" | "luxury_watch" | "yacht" | "private_jet" | "other" | "not_lead",
  "lead_type": "buyer" | "broker" | "not_lead",
  "market": "Dubai/Qatar/Spain/..." or "",
  "confidence": number between 0 and 1,
  "reason": "short explanation",
  "evidence": "short excerpt proving intent"
}
""".strip()

    base_url = (os.getenv("PERPLEXITY_BASE_URL") or "https://api.perplexity.ai").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.perplexity_api_key}", "Content-Type": "application/json"}
    user_payload = f"""
{master_prompt}

URL: {url}
""".strip()

    payload = {
        "model": cfg.perplexity_model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON (no markdown, no commentary)."},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.2,
        "web_search_options": {
            "search_context_size": "high"
        },
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data_json = r.json()
    content = (((data_json.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    data = _extract_json_loose(content)
    data.setdefault("is_lead", False)
    data.setdefault("vertical", "not_lead")
    data.setdefault("lead_type", "not_lead")
    data.setdefault("market", "")
    data.setdefault("confidence", 0.0)
    data.setdefault("reason", "")
    data.setdefault("evidence", "")
    return data


def _validate_verdict(cfg: Config, verdict: dict, source_text: str, market_hint: str = "") -> dict:
    v = dict(verdict or {})

    is_lead = bool(v.get("is_lead"))
    source = str(v.get("source") or "op")
    buyer_stage = str(v.get("buyer_stage") or "researching")
    property_type = str(v.get("property_type") or "unknown")
    budget = str(v.get("budget") or "not_mentioned")
    reason = str(v.get("reason") or "")
    evidence = str(v.get("evidence") or "")
    thread_type = str(v.get("thread_type") or "general_discussion")
    thread_summary = str(v.get("thread_summary") or "")

    try:
        confidence = float(v.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    src = (source_text or "").strip()
    src_lower = src.lower()

    # If the model provides "evidence", check that key words are present (relaxed check)
    # LLMs often paraphrase, so we check for significant words rather than exact match
    ev = (evidence or "").strip()
    if ev:
        # Extract significant words (4+ chars) from evidence
        ev_words = [w.lower() for w in re.findall(r'\b\w{4,}\b', ev)]
        # Require at least 50% of significant words to be present in source
        if ev_words:
            matches = sum(1 for w in ev_words if w in src_lower)
            match_ratio = matches / len(ev_words)
            if match_ratio < 0.4:  # Less than 40% of words found = likely hallucination
                return {
                    "is_lead": False,
                    "source": source,
                    "buyer_stage": "researching",
                    "confidence": 0.0,
                    "property_type": "unknown",
                    "budget": "not_mentioned",
                    "evidence": "",
                    "reason": "Forced not_lead: evidence not found in input context.",
                    "thread_type": "general_discussion",
                    "thread_summary": thread_summary,
                }

    # Prevent obvious false positives: require real estate hints if a lead is claimed.
    if is_lead:
        real_estate_hints = _VERTICAL_HINTS.get("real_estate", [])
        has_hint = any(h in src_lower for h in real_estate_hints)
        if not has_hint and "/comments/" not in src_lower:
            return {
                "is_lead": False,
                "source": source,
                "buyer_stage": "researching",
                "confidence": 0.0,
                "property_type": "unknown",
                "budget": "not_mentioned",
                "evidence": "",
                "reason": "Forced not_lead: no real estate evidence in provided context.",
                "thread_type": "general_discussion",
                "thread_summary": thread_summary,
            }

    # Exclude renters - check for rental keywords
    rental_tokens = [" rent", " rental", " lease", " leasing", " charter", " short term"]
    if is_lead and any(tok in f" {src_lower} " for tok in rental_tokens):
        # Check if it's primarily about renting vs buying
        buy_tokens = [" buy", " purchase", " buying", " purchasing"]
        has_buy = any(tok in f" {src_lower} " for tok in buy_tokens)
        if not has_buy:
            is_lead = False
            confidence = min(confidence, 0.25)
            reason = "Excluded: appears to be about renting, not buying."

    return {
        "is_lead": bool(is_lead),
        "source": source,
        "buyer_stage": buyer_stage,
        "confidence": confidence,
        "property_type": property_type,
        "budget": budget,
        "evidence": evidence,
        "reason": reason,
        "thread_type": thread_type,
        "thread_summary": thread_summary,
    }


def _classifier_label(cfg: Config) -> str:
    if cfg.openai_api_key:
        return f"openai:{cfg.openai_model}"
    if cfg.perplexity_api_key:
        return f"perplexity:{cfg.perplexity_model}"
    if cfg.gemini_api_key:
        return f"gemini:{cfg.gemini_model}"
    return "heuristic"


_INTENT_WORDS = {
    "buyer": ["looking to buy", "buying", "purchase", "buy", "budget", "mortgage", "lease", "financing"],
    "seller": ["selling", "sell", "listing", "for sale", "offloading"],
    "renter": ["rent", "rental", "lease", "tenant", "short term"],
    "broker": ["recommend", "recommendation", "agent", "broker", "dealer", "where to buy", "who to contact", "ad"],
}

_VERTICAL_HINTS = {
    "luxury_car": ["ferrari", "lamborghini", "rolls", "rolls-royce", "bugatti", "mclaren", "supercar", "dealer"],
    "luxury_watch": ["rolex", "patek", "audemars", "ap", "richard mille", "nautilus", "daytona", "submariner", "ad"],
    "yacht": ["yacht", "superyacht", "charter", "marina berth", "berth"],
    "private_jet": ["private jet", "gulfstream", "netjets", "fractional", "jet charter"],
    "real_estate": ["apartment", "villa", "property", "real estate", "broker", "agent", "penthouse", "off plan"],
}


def _prefilter_batch(cfg: Config, items: list[dict], status_cb: Callable[..., None] | None = None) -> tuple[list[dict], list[dict]]:
    """
    Pre-filter hits using title/snippet before expensive Apify scraping.
    Returns (certain_leads, needs_verification):
    - certain_leads: High confidence leads from title/snippet alone
    - needs_verification: Need to scrape full content to verify
    """
    if not cfg.openai_api_key:
        # No OpenAI key - return all items as needing verification
        return [], items

    if not items:
        return [], []

    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.openai_api_key,
        organization=cfg.openai_organization,
        project=cfg.openai_project,
        timeout=60,
        max_retries=2,
    )

    # Process in batches of 20
    batch_size = 20
    certain_leads: list[dict] = []
    needs_verification: list[dict] = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        # Build prompt with all items in batch
        items_text = ""
        for idx, item in enumerate(batch):
            title = item.get("title", "")[:200]
            snippet = item.get("snippet", "")[:300]
            items_text += f"\n{idx + 1}. Title: {title}\n   Snippet: {snippet}\n"

        prompt = f"""You are a lead classifier for REAL ESTATE only. Review these Reddit post titles/snippets.

ONLY classify as leads if about REAL ESTATE (buying/selling property, apartments, villas, houses).
EXCLUDE: renters, tenants, people looking to rent/lease. We only want BUYERS and SELLERS.
EXCLUDE: luxury cars, watches, yachts, jets - NOT relevant.

For each item, classify as:
- "certain_lead": Clear BUYING or SELLING property intent (e.g., "Looking to buy apartment", "Selling my villa")
- "verify": Might be real estate buyer/seller but need full content to confirm
- "not_lead": Renters, other topics, news, memes, complaints

Items:
{items_text}

Return JSON:
{{
  "certain_leads": [{{"id": 1, "vertical": "real_estate", "lead_type": "buyer", "market": "Dubai"}}],
  "verify": [2, 5],
  "not_leads": [3, 4]
}}

For certain_leads, include: id, vertical (always "real_estate"), lead_type ("buyer" or "seller"), market (city/country).
Be strict - only BUYING or SELLING property. Renters = not_lead."""

        try:
            resp = client.chat.completions.create(
                model=cfg.openai_model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            data = _extract_json_loose(text)

            for entry in data.get("certain_leads", []):
                # Handle both formats: {"id": 1, "vertical": ...} or just 1
                if isinstance(entry, dict):
                    idx = entry.get("id")
                    if isinstance(idx, int) and 1 <= idx <= len(batch):
                        item = batch[idx - 1].copy()
                        item["prefilter_status"] = "certain_lead"
                        item["vertical"] = entry.get("vertical", "real_estate")
                        item["lead_type"] = entry.get("lead_type", "buyer")
                        item["market"] = entry.get("market", "")
                        certain_leads.append(item)
                elif isinstance(entry, int) and 1 <= entry <= len(batch):
                    item = batch[entry - 1].copy()
                    item["prefilter_status"] = "certain_lead"
                    item["vertical"] = "real_estate"
                    item["lead_type"] = "buyer"
                    certain_leads.append(item)

            for idx in data.get("verify", []):
                if isinstance(idx, int) and 1 <= idx <= len(batch):
                    item = batch[idx - 1].copy()
                    item["prefilter_status"] = "verify"
                    needs_verification.append(item)

        except Exception as e:
            # On error, mark all for verification (fail open)
            if status_cb:
                status_cb(stage="prefilter", stage_detail=f"Prefilter error: {type(e).__name__}, marking all for verification")
            needs_verification.extend(batch)

        if status_cb:
            status_cb(stage="prefilter", stage_detail=f"Pre-filtered {min(i + batch_size, len(items))}/{len(items)}: {len(certain_leads)} certain, {len(needs_verification)} to verify")

        time.sleep(0.5)  # Rate limit

    return certain_leads, needs_verification


def _heuristic_classify(text: str, market_hint: str = "") -> dict:
    t = (text or "").lower()

    # Check for real estate hints
    has_real_estate = any(h in t for h in _VERTICAL_HINTS.get("real_estate", []))

    # Check for buyer intent
    buyer_words = ["looking to buy", "buying", "purchase", "buy", "budget", "mortgage", "financing"]
    has_buyer_intent = any(w in t for w in buyer_words)

    is_lead = has_real_estate and has_buyer_intent

    # Determine buyer stage based on keywords
    buyer_stage = "researching"
    if any(w in t for w in ["ready to", "about to", "going to buy", "closing", "making offer"]):
        buyer_stage = "ready_to_buy"
    elif any(w in t for w in ["looking for", "searching", "visiting", "viewing"]):
        buyer_stage = "actively_looking"

    # Try to detect property type
    property_type = "unknown"
    if "apartment" in t or "flat" in t:
        property_type = "apartment"
    elif "villa" in t:
        property_type = "villa"
    elif "house" in t or "home" in t:
        property_type = "house"
    elif "townhome" in t or "townhouse" in t:
        property_type = "townhome"

    return {
        "is_lead": bool(is_lead),
        "source": "op",
        "buyer_stage": buyer_stage if is_lead else "researching",
        "confidence": 0.65 if is_lead else 0.25,
        "property_type": property_type,
        "budget": "not_mentioned",
        "evidence": _truncate(text, 280),
        "reason": "Heuristic classification (no API key configured)." if is_lead else "No clear buying intent found.",
        "thread_type": "buyer_seeking" if is_lead else "general_discussion",
        "thread_summary": "",
    }


def _market_hint_from_url_or_query(url: str, query: str) -> str:
    u = (url or "").lower()
    q = (query or "").lower()
    for token, market in [
        ("/r/dubai", "Dubai"),
        ("/r/abudhabi", "Abu Dhabi"),
        ("/r/qatar", "Qatar"),
        ("/r/saudi", "Saudi"),
        ("/r/london", "London"),
        ("/r/miami", "Miami"),
        ("/r/losangeles", "Los Angeles"),
        ("/r/nyc", "New York"),
        ("/r/spain", "Spain"),
        ("costa del sol", "Spain"),
        ("marbella", "Spain"),
        ("barcelona", "Spain"),
        ("madrid", "Spain"),
        ("monaco", "Monaco"),
    ]:
        if token in u or token in q:
            return market
    return ""


def run_once(
    cfg: Config | None = None,
    status_cb: Callable[..., None] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    ensure_dir(cfg.data_dir)

    queries = get_queries()
    hits_csv = _hits_csv_path(cfg)
    if cfg.use_hits_cache and _is_recent_file(hits_csv, cfg.hits_max_age_hours):
        if status_cb:
            status_cb(stage="firecrawl", stage_detail=f"Using cached hits: {hits_csv}")
    else:
        if status_cb:
            status_cb(stage="firecrawl", stage_detail=f"Firecrawl: {len(queries)} queries (tbs={cfg.tbs})")
        hits_csv = firecrawl_collect_hits(cfg, queries)
        if status_cb:
            status_cb(stage="firecrawl", stage_detail=f"Firecrawl done: {hits_csv}")
    urls = _read_unique_urls_from_hits_csv(hits_csv)
    before_filter = len(urls)
    urls = [u for u in urls if _is_reddit_post_or_comment_url(u) and _is_allowed_subreddit(u) and not _is_excluded_reddit_url(u)]
    if cfg.max_urls > 0 and len(urls) > cfg.max_urls:
        urls = urls[: cfg.max_urls]
    if status_cb:
        dropped = before_filter - len(urls)
        status_cb(stage="urls", stage_detail=f"Using {len(urls)} post/comment URLs (dropped {dropped} excluded/non-post)")

    url_meta: dict[str, dict] = {}
    with hits_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = _normalize_reddit_url(row.get("url", ""))
            if not u or not _is_reddit_post_or_comment_url(u) or _is_excluded_reddit_url(u):
                continue
            url_meta.setdefault(
                u,
                {"url": u, "query": row.get("query", ""), "title": row.get("title", ""), "snippet": row.get("snippet", "")},
            )

    # Pre-filter using title/snippet before expensive Apify scraping
    if status_cb:
        status_cb(stage="prefilter", stage_detail=f"Pre-filtering {len(urls)} URLs with OpenAI...")

    prefilter_items = [url_meta.get(u, {"url": u, "title": "", "snippet": ""}) for u in urls]
    certain_leads, needs_verification = _prefilter_batch(cfg, prefilter_items, status_cb=status_cb)

    if status_cb:
        status_cb(stage="prefilter", stage_detail=f"Pre-filter done: {len(certain_leads)} likely leads, {len(needs_verification)} to verify, {len(urls) - len(certain_leads) - len(needs_verification)} discarded")

    # FIXED: ALL leads must go through full content validation - no shortcuts!
    # Combine certain_leads and needs_verification for full Apify scraping
    all_leads_to_validate = certain_leads + needs_verification
    max_verify = 40  # Increased limit since we're validating all potential leads
    urls_to_scrape = [item.get("url") for item in all_leads_to_validate if item.get("url")][:max_verify]
    if len(all_leads_to_validate) > max_verify:
        if status_cb:
            status_cb(stage="prefilter", stage_detail=f"Limiting validation to {max_verify} URLs (had {len(all_leads_to_validate)})")

    all_out = cfg.data_dir / "all_results.csv"
    leads_out = cfg.data_dir / "leads_only.csv"
    fieldnames = [
        "url",
        "source",
        "buyer_stage",
        "confidence",
        "property_type",
        "budget",
        "evidence",
        "reason",
        "thread_type",
        "thread_summary",
    ]

    total_count = 0
    leads_count = 0

    # Stream writes so the UI/CSV updates during a run.
    with all_out.open("w", newline="", encoding="utf-8") as all_f, leads_out.open("w", newline="", encoding="utf-8") as leads_f:
        all_w = csv.DictWriter(all_f, fieldnames=fieldnames)
        leads_w = csv.DictWriter(leads_f, fieldnames=fieldnames)
        all_w.writeheader()
        leads_w.writeheader()
        all_f.flush()
        leads_f.flush()

        def _write_row(row: dict, is_lead: bool) -> None:
            nonlocal total_count, leads_count
            all_w.writerow(row)
            all_f.flush()
            total_count += 1
            if is_lead:
                leads_w.writerow(row)
                leads_f.flush()
                leads_count += 1

        # ============================================================
        # Step 1: Scrape ALL potential leads with Apify for full validation
        # (No more direct writes - all leads must be validated with full content)
        # ============================================================
        if urls_to_scrape:
            if status_cb:
                status_cb(stage="apify", stage_detail=f"Scraping {len(urls_to_scrape)} URLs with Apify...")

            all_items: list[dict] = []
            for batch_urls in _chunks(urls_to_scrape, cfg.apify_max_urls):
                if status_cb:
                    status_cb(stage="apify", stage_detail=f"Apify batch: {len(batch_urls)} URLs (total scraped: {len(all_items)} items)")
                items = _run_apify_batch(cfg, batch_urls, status_cb=status_cb)
                all_items.extend(items)
        else:
            all_items = []

        posts, comments_by_post = _group_items(all_items)

        if status_cb:
            status_cb(stage="apify", stage_detail=f"Apify done: {len(posts)} posts, {sum(len(c) for c in comments_by_post.values())} comments")

        # Step 2: Classify each post using OpenAI with FULL thread content (the only path now)
        if status_cb:
            status_cb(stage="classify", stage_detail=f"Classifying {len(posts)} posts (classifier={_classifier_label(cfg)})")

        completed = 0
        for pid, post in posts.items():
            url = post.get("url") or post.get("link") or ""
            title = post.get("title", "")
            meta = url_meta.get(_normalize_reddit_url(url), {})
            query = meta.get("query", "")
            market_hint = _market_hint_from_url_or_query(url, query)

            # Build thread text from post + comments
            comments = comments_by_post.get(pid, [])
            thread_text = _build_thread_text(cfg, post, comments)

            try:
                if cfg.openai_api_key:
                    verdict = _openai_classify(cfg, thread_text)
                elif cfg.gemini_api_key:
                    verdict = _gemini_classify(cfg, thread_text)
                else:
                    verdict = _heuristic_classify(thread_text, market_hint=market_hint)
            except Exception as e:
                verdict = {
                    "is_lead": False,
                    "vertical": "not_lead",
                    "lead_type": "not_lead",
                    "market": market_hint,
                    "confidence": 0.0,
                    "reason": f"Classifier error: {type(e).__name__}: {e}",
                    "evidence": "",
                }

            # Validate verdict against thread text
            verdict = _validate_verdict(cfg, verdict, thread_text, market_hint=market_hint)

            _write_row(
                {
                    "url": url,
                    "source": verdict.get("source", "op"),
                    "buyer_stage": verdict.get("buyer_stage", "researching"),
                    "confidence": verdict.get("confidence", 0.0),
                    "property_type": verdict.get("property_type", "unknown"),
                    "budget": verdict.get("budget", "not_mentioned"),
                    "evidence": verdict.get("evidence", ""),
                    "reason": verdict.get("reason", ""),
                    "thread_type": verdict.get("thread_type", "general_discussion"),
                    "thread_summary": verdict.get("thread_summary", title),
                },
                is_lead=bool(verdict.get("is_lead")),
            )

            completed += 1
            if status_cb and (completed % 5 == 0 or completed == len(posts)):
                status_cb(
                    stage="classify",
                    stage_detail=f"Classified {completed}/{len(posts)} posts (classifier={_classifier_label(cfg)})",
                )
            time.sleep(cfg.sleep_between_llm_calls_sec)

        if status_cb:
            status_cb(stage="write", stage_detail="Writing CSV outputs…")

        return {
            "hits_csv": str(hits_csv),
            "all_out": str(all_out),
            "leads_out": str(leads_out),
            "leads": leads_count,
            "total": total_count,
        }
