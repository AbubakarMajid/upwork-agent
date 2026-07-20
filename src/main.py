"""Streaming orchestration loop.

Each cycle: incremental concurrent fetch -> submit one independent pipeline task
per new job. A task flows LLM scope filter -> browser fetch (pooled, captcha
serialized) -> code metric scoring -> proposal draft -> Discord. Jobs run
concurrently and a job that clears scoring is sent the instant it's ready - the
loop never waits for the whole batch.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from browser_fetcher import BrowserPool
from db import JobStore, NullJobStore
from job_fetcher import JobFetcher
from notifier_discord import DiscordNotifier
from proposal_drafter import draft_proposal
from scope_filter import scope_filter_job
from scorer import score_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upwork_agent")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _process_job(job, config, store, browser, notifier, llm_client) -> str:
    """Run one job end-to-end. Returns a short status string for logging."""
    # 1. LLM scope filter (cheap fast-pass data, before any browser work)
    verdict = scope_filter_job(job, config, client=llm_client)
    if not verdict.relevant:
        store.mark_rejected(job.id, job.title)
        return f"scope-rejected: {job.title} -> {verdict.reason}"

    # 2. Browser fetch (pooled; captcha serialized). On failure, mark seen so we
    #    don't retry a job that keeps failing.
    try:
        details = browser.fetch(job.id)
    except Exception:
        log.exception(f"browser fetch failed for job {job.id}")
        store.mark_seen(job.id, job.title)
        return f"browser-failed: {job.title}"

    # 3. Code-based metric scoring (pay rate / proposals / hires / country / constraints)
    result = score_job(job, details, verdict.matched_areas or job.matched_keywords, config)
    if not result.passed:
        store.mark_rejected(job.id, job.title)
        return f"score-rejected: {job.title} -> {result.reasons}"

    # 4. Draft proposal
    try:
        proposal = draft_proposal(job, details, config, client=llm_client)
    except Exception:
        log.exception(f"proposal drafting failed for job {job.id}")
        store.mark_seen(job.id, job.title)
        return f"draft-failed: {job.title}"

    # 5. Notify
    notifier.send_job_alert(job, details, result, proposal)
    store.mark_notified(job.id, job.title)
    return f"notified: {job.title} (score={result.score})"


def run_once(config, store, fetcher, browser, notifier, llm_client, executor, keywords: list[str]) -> int:
    lookback = int(config.get("fetch_lookback_minutes", 30))
    jobs = fetcher.fetch_new(keywords, store, lookback_minutes=lookback)
    for job in jobs:
        store.mark_seen(job.id, job.title, publish_time=job.publish_time)
    log.info(f"fetched {len(jobs)} new jobs across {len(keywords)} keywords")

    notified = 0
    futures = {
        executor.submit(_process_job, job, config, store, browser, notifier, llm_client): job
        for job in jobs
    }
    for fut in as_completed(futures):
        try:
            status = fut.result()
            log.info(status)
            if status.startswith("notified:"):
                notified += 1
        except Exception:
            log.exception(f"pipeline error for job {futures[fut].id}")
    return notified


def parse_args():
    parser = argparse.ArgumentParser(description="Upwork job-alert agent")
    parser.add_argument(
        "--keyword", action="append", default=None,
        help="Search keyword to use instead of config.yaml's search_keywords. Repeatable.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit instead of looping forever.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    config = load_config()
    keywords = args.keyword if args.keyword else config["search_keywords"]

    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", 300))
    pool_size = int(config.get("browser_pool_size", 2))
    max_workers = int(config.get("max_pipeline_concurrency", 6))
    headless = bool(config.get("browser_headless", False))
    xvfb = bool(config.get("browser_xvfb", False))
    captcha_lock_timeout = float(config.get("captcha_lock_timeout_seconds", 60))
    panel_wait_timeout = float(config.get("client_panel_wait_seconds", 20))

    # PERSIST_JOBS=0 swaps in a no-op store so nothing is recorded and every run
    # re-processes the same jobs - needed for debugging, since Upwork serves a job
    # to the anonymous fetcher only once. Defaults to persisting.
    persist_jobs = os.environ.get("PERSIST_JOBS", "1").lower() not in ("0", "false", "no")
    store = JobStore() if persist_jobs else NullJobStore()
    if not persist_jobs:
        log.warning("PERSIST_JOBS disabled - jobs are NOT being recorded (dev mode; every run re-processes all jobs)")
    fetcher = JobFetcher()
    notifier = DiscordNotifier()
    llm_client = OpenAI()  # reads OPENAI_API_KEY

    with BrowserPool(
        size=pool_size, headless=headless, xvfb=xvfb,
        captcha_lock_timeout=captcha_lock_timeout, panel_wait_timeout=panel_wait_timeout,
    ) as browser, \
            ThreadPoolExecutor(max_workers=max_workers) as executor:
        if args.once:
            log.info(f"single-cycle test run, keywords={keywords}")
            notified = run_once(config, store, fetcher, browser, notifier, llm_client, executor, keywords)
            log.info(f"done - {notified} job(s) notified")
            return

        while True:
            try:
                run_once(config, store, fetcher, browser, notifier, llm_client, executor, keywords)
            except Exception:
                log.exception("error in poll cycle")
            time.sleep(interval)


if __name__ == "__main__":
    main()
