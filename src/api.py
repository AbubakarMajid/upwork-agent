"""FastAPI layer over the Upwork job-alert agent.

Wraps the existing pipeline (main.run_once) behind an HTTP API instead of the
CLI polling loop. The expensive shared resources - the Chrome browser pool, the
job store, the LLM client and the pipeline thread pool - are built once on
startup (lifespan) and reused across requests, mirroring how main.main() holds
them for the lifetime of the process.

Endpoints:
  GET  /health          - liveness + resource/store summary
  GET  /config          - effective config (keywords + scoring gates)
  GET  /jobs            - recorded jobs from the SQLite store (filter by status)
  POST /run             - run one pipeline cycle now (blocking) and return results
  POST /run/background  - kick off one cycle in the background, return immediately

Run with:  uvicorn api:app --host 0.0.0.0 --port 8000   (from src/, or api:app on path)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel

from browser_fetcher import BrowserPool
from db import JobStore, NullJobStore
from job_fetcher import JobFetcher
from main import load_config, run_once
from notifier_discord import DiscordNotifier

log = logging.getLogger("upwork_agent.api")


@dataclass
class Engine:
    """Long-lived pipeline resources, shared across requests."""
    config: dict
    store: object
    fetcher: JobFetcher
    browser: BrowserPool
    notifier: DiscordNotifier
    llm_client: OpenAI
    executor: ThreadPoolExecutor
    # Serializes /run cycles - run_once shares the browser pool + store and is not
    # meant to be entered concurrently.
    run_lock: threading.Lock

    def default_keywords(self) -> list[str]:
        return list(self.config["search_keywords"])

    def run_cycle(self, keywords: list[str]) -> int:
        with self.run_lock:
            return run_once(
                self.config, self.store, self.fetcher, self.browser,
                self.notifier, self.llm_client, self.executor, keywords,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    config = load_config()

    pool_size = int(config.get("browser_pool_size", 2))
    max_workers = int(config.get("max_pipeline_concurrency", 6))
    headless = bool(config.get("browser_headless", False))
    xvfb = bool(config.get("browser_xvfb", False))
    captcha_lock_timeout = float(config.get("captcha_lock_timeout_seconds", 60))
    panel_wait_timeout = float(config.get("client_panel_wait_seconds", 20))

    persist_jobs = os.environ.get("PERSIST_JOBS", "1").lower() not in ("0", "false", "no")
    store = JobStore() if persist_jobs else NullJobStore()
    if not persist_jobs:
        log.warning("PERSIST_JOBS disabled - jobs are NOT being recorded (dev mode)")

    browser = BrowserPool(
        size=pool_size, headless=headless, xvfb=xvfb,
        captcha_lock_timeout=captcha_lock_timeout, panel_wait_timeout=panel_wait_timeout,
    )
    browser.__enter__()  # BrowserPool is a context manager; open it for the app's lifetime
    executor = ThreadPoolExecutor(max_workers=max_workers)

    app.state.engine = Engine(
        config=config,
        store=store,
        fetcher=JobFetcher(),
        browser=browser,
        notifier=DiscordNotifier(),
        llm_client=OpenAI(),  # reads OPENAI_API_KEY
        executor=executor,
        run_lock=threading.Lock(),
    )
    log.info("engine ready (pool_size=%d, workers=%d)", pool_size, max_workers)
    try:
        yield
    finally:
        executor.shutdown(wait=False)
        browser.__exit__(None, None, None)
        store.close()
        log.info("engine shut down")


app = FastAPI(
    title="Upwork Job-Alert Agent",
    version="1.0.0",
    description="HTTP layer over the no-login Upwork scraping + scoring + Discord pipeline.",
    lifespan=lifespan,
)


def _engine() -> Engine:
    return app.state.engine


class RunRequest(BaseModel):
    # Override config.yaml's search_keywords for this cycle only. Omit to use config.
    keywords: list[str] | None = None


class RunResult(BaseModel):
    notified: int
    keywords: list[str]


@app.get("/health")
def health():
    eng = _engine()
    return {
        "status": "ok",
        "browser_pool_size": eng.config.get("browser_pool_size"),
        "store": type(eng.store).__name__,
        "job_counts": eng.store.counts_by_status(),
    }


@app.get("/config")
def get_config():
    eng = _engine()
    return {
        "search_keywords": eng.config.get("search_keywords"),
        "scoring": eng.config.get("scoring"),
        "fetch_lookback_minutes": eng.config.get("fetch_lookback_minutes"),
    }


@app.get("/jobs")
def list_jobs(
    status: str | None = Query(None, description="Filter: seen | rejected | notified"),
    limit: int = Query(100, ge=1, le=1000),
):
    return {"jobs": _engine().store.list_jobs(status=status, limit=limit)}


@app.post("/run", response_model=RunResult)
async def run(req: RunRequest | None = None):
    """Run one pipeline cycle now and return how many jobs were notified.

    Blocking: a cycle fetches, filters, browser-scrapes and scores jobs, so this
    can take a while. Runs in a worker thread so the event loop stays responsive.
    """
    eng = _engine()
    keywords = (req.keywords if req and req.keywords else None) or eng.default_keywords()
    if eng.run_lock.locked():
        raise HTTPException(status_code=409, detail="a run cycle is already in progress")
    try:
        notified = await asyncio.to_thread(eng.run_cycle, keywords)
    except Exception as e:
        log.exception("run cycle failed")
        raise HTTPException(status_code=500, detail=str(e))
    return RunResult(notified=notified, keywords=keywords)


@app.post("/run/background", status_code=202)
async def run_background(req: RunRequest | None = None):
    """Kick off one cycle in the background and return immediately."""
    eng = _engine()
    keywords = (req.keywords if req and req.keywords else None) or eng.default_keywords()
    if eng.run_lock.locked():
        raise HTTPException(status_code=409, detail="a run cycle is already in progress")

    def _bg():
        try:
            eng.run_cycle(keywords)
        except Exception:
            log.exception("background run cycle failed")

    threading.Thread(target=_bg, name="run-cycle", daemon=True).start()
    return {"status": "started", "keywords": keywords}
