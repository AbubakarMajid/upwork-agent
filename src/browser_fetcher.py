"""Slow-pass fetcher: opens a job's detail page anonymously (no Upwork login),
clears the Cloudflare Turnstile challenge with a real simulated click, and
extracts apply link, client info, and proposal/activity data.

Only call this for jobs that already passed the LLM scope filter (scope_filter.py) -
each call opens a real Chrome instance and is much slower than job_fetcher.py.

Headless note: the Cloudflare Turnstile solve (uc_gui_click_captcha) drives the real
OS mouse via PyAutoGUI, so it needs a display. On Linux, run headless with xvfb=True
(a virtual display) and the captcha click still works. On macOS there is no xvfb, so
headless=True leaves the captcha-click with no screen to act on - run visible there
(only `browser_pool_size` windows ever open, reused across cycles), or run the whole
agent in a Linux container.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from seleniumbase import SB

log = logging.getLogger("upwork_agent.browser")

_DEBUG_DIR = Path(__file__).resolve().parent.parent / "resources" / "debug"

# How many times to (re)load a job page before giving up. Cloudflare sometimes
# clears the challenge but wedges on the "Verification successful. Waiting for
# www.upwork.com to respond" interstitial; re-opening the URL usually unsticks it.
_MAX_LOAD_ATTEMPTS = 2


def _dump_debug(html: str, job_id: str) -> None:
    """DEBUG_SCRAPE=1: save the captured HTML and log which client markers it
    contains, so a missing client panel can be diagnosed from the real source."""
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = _DEBUG_DIR / f"{job_id}.html"
    path.write_text(html, encoding="utf-8")
    markers = ["client-hires", "client-contract-date", "client-location", "client-spend", "client-job-posting-stats"]
    present = {m: (m in html) for m in markers}
    log.warning("DEBUG_SCRAPE job %s: html_len=%d saved=%s markers=%s", job_id, len(html), path, present)


@dataclass
class JobDetails:
    job_id: str
    url: str
    budget_type: str | None = None  # "Hourly" or "Fixed-price"
    hourly_min: str | None = None
    hourly_max: str | None = None
    fixed_price: str | None = None
    proposals: str | None = None
    last_viewed_by_client: str | None = None
    interviewing: str | None = None
    invites_sent: str | None = None
    unanswered_invites: str | None = None
    job_hires: str | None = None  # hires on THIS posting (activity section), not the client's lifetime total
    client_member_since: str | None = None
    client_location: str | None = None
    client_total_spent: str | None = None
    client_hires: str | None = None
    client_jobs_posted: str | None = None
    client_hours: str | None = None
    client_industry: str | None = None
    country_restricted_note: str | None = None


def _job_url(job_id: str) -> str:
    return f"https://www.upwork.com/jobs/x_~02{job_id}/"


_PROPOSAL_TIER_RE = re.compile(r"(less than \d+|\d+\s*to\s*\d+|\d+\+|\d+)\s*proposals?", re.I)
_HOURLY_RANGE_RE = re.compile(r"\$([\d,.]+)\s*-\s*\$([\d,.]+)\s*/\s*hr", re.I)
_HOURLY_SINGLE_RE = re.compile(r"\$([\d,.]+)\s*/\s*hr", re.I)
_FIXED_PRICE_RE = re.compile(r"fixed[\s-]price[^$]{0,40}\$([\d,.]+)", re.I)
_JOBS_POSTED_RE = re.compile(r"(\d+)\s*jobs?\s*posted", re.I)


def _extract_budget(soup: BeautifulSoup, page_text: str, details: JobDetails) -> None:
    """Structured selector first; fall back to regex over the full page text if
    Upwork's markup doesn't match (budget is the single most important field)."""
    for li in soup.find_all("li"):
        desc = li.select_one("div.description")
        if not desc:
            continue
        label = desc.get_text(strip=True)
        if label not in ("Hourly", "Fixed-price", "Fixed price"):
            continue
        amounts = [s.get_text(strip=True) for s in li.select("strong") if s.get_text(strip=True).startswith("$")]
        if not amounts:
            continue
        details.budget_type = "Hourly" if label == "Hourly" else "Fixed-price"
        if label == "Hourly" and len(amounts) >= 2:
            details.hourly_min, details.hourly_max = amounts[0], amounts[1]
        elif label == "Hourly" and len(amounts) == 1:
            details.hourly_min = details.hourly_max = amounts[0]
        else:
            details.fixed_price = amounts[0]
        return

    # Fallback: regex over the rendered text.
    range_match = _HOURLY_RANGE_RE.search(page_text)
    if range_match:
        details.budget_type = "Hourly"
        details.hourly_min = f"${range_match.group(1)}"
        details.hourly_max = f"${range_match.group(2)}"
        return

    single_match = _HOURLY_SINGLE_RE.search(page_text)
    if single_match:
        details.budget_type = "Hourly"
        details.hourly_min = details.hourly_max = f"${single_match.group(1)}"
        return

    fixed_match = _FIXED_PRICE_RE.search(page_text)
    if fixed_match:
        details.budget_type = "Fixed-price"
        details.fixed_price = f"${fixed_match.group(1)}"


def _extract(html: str, job_id: str, url: str) -> JobDetails:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    details = JobDetails(job_id=job_id, url=url)

    _extract_budget(soup, page_text, details)

    activity = {}
    for li in soup.select("ul.client-activity-items li.ca-item"):
        title_el = li.select_one("span.title")
        if not title_el:
            continue
        key = title_el.get_text(strip=True).rstrip(":")
        value_el = li.select_one("span.value, div.value")
        activity[key] = value_el.get_text(strip=True) if value_el else None

    details.proposals = activity.get("Proposals")
    if not details.proposals:
        tier_match = _PROPOSAL_TIER_RE.search(page_text)
        if tier_match:
            details.proposals = tier_match.group(1).strip().title()

    details.last_viewed_by_client = activity.get("Last viewed by client")
    details.interviewing = activity.get("Interviewing")
    details.invites_sent = activity.get("Invites sent")
    details.unanswered_invites = activity.get("Unanswered invites")
    # Hires on THIS posting - Upwork omits the row when it's zero, so absent means 0.
    details.job_hires = activity.get("Hires") or activity.get("Hired")

    contract_date = soup.select_one('[data-qa="client-contract-date"]')
    details.client_member_since = contract_date.get_text(strip=True) if contract_date else None

    location = soup.select_one('[data-qa="client-location"] strong')
    details.client_location = location.get_text(strip=True) if location else None

    spend = soup.select_one('[data-qa="client-spend"]')
    details.client_total_spent = spend.get_text(strip=True) if spend else None

    hires = soup.select_one('[data-qa="client-hires"]')
    details.client_hires = hires.get_text(strip=True) if hires else None

    # No confirmed data-qa selector for "jobs posted" - try a few likely
    # candidates, then fall back to regex over the page text.
    for selector in ('[data-qa="client-job-posting-stats"]', '[data-qa="client-jobs-posted"]'):
        jobs_posted_el = soup.select_one(selector)
        if jobs_posted_el:
            details.client_jobs_posted = jobs_posted_el.get_text(strip=True)
            break
    if not details.client_jobs_posted:
        jobs_posted_match = _JOBS_POSTED_RE.search(page_text)
        if jobs_posted_match:
            details.client_jobs_posted = f"{jobs_posted_match.group(1)} jobs posted"

    hours = soup.select_one('[data-qa="client-hours"]')
    details.client_hours = hours.get_text(strip=True) if hours else None

    industry = soup.select_one('[data-qa="client-company-profile-industry"]')
    details.client_industry = industry.get_text(strip=True) if industry else None

    restriction = soup.find(string=re.compile(r"may apply", re.I))
    details.country_restricted_note = restriction.strip() if restriction else None

    return details


class BrowserFetcher:
    """Reusable anonymous browser session for the slow pass.

    `captcha_lock` serializes the Cloudflare Turnstile click across instances:
    uc_gui_click_captcha() drives the real OS mouse cursor (PyAutoGUI), and on a
    single display only one click can happen at a time. Page open + scrape run
    unlocked, so they overlap across pooled instances.

    The lock is acquired with `captcha_lock_timeout` rather than blocking forever:
    uc_gui_click_captcha() can itself hang indefinitely (e.g. another pooled Chrome
    window steals OS focus mid-click, or the captcha widget never renders), and with
    no timeout that wedges every other thread waiting on this same lock - the whole
    pipeline looks "stuck" with no error and no crash. A timeout turns that into a
    single failed fetch instead.
    """

    def __init__(
        self,
        headless: bool = False,
        xvfb: bool = False,
        captcha_lock: threading.Lock | None = None,
        captcha_lock_timeout: float = 60.0,
        panel_wait_timeout: float = 20.0,
    ):
        self._headless = headless
        self._xvfb = xvfb
        self._sb = None
        self._captcha_lock = captcha_lock or threading.Lock()
        self._captcha_lock_timeout = captcha_lock_timeout
        self._panel_wait_timeout = panel_wait_timeout

    def __enter__(self):
        # xvfb (Linux virtual display) lets the GUI captcha click work without a
        # visible window. On macOS there is no xvfb - headless there means the
        # captcha-click step has no screen to click on (see module note).
        sb_kwargs = dict(uc=True, test=True, headless=self._headless, xvfb=self._xvfb)
        # CHROME_BINARY_PATH lets a host use a system Chromium instead of the one
        # SeleniumBase auto-downloads - needed on arm64 Linux, where Google ships
        # no linux-arm64 chromedriver so the auto-download path can't run. Unset on
        # x86_64 (the intended deploy), so default behavior is unchanged there.
        chrome_binary = os.environ.get("CHROME_BINARY_PATH")
        if chrome_binary:
            sb_kwargs["binary_location"] = chrome_binary
        self._ctx = SB(**sb_kwargs)
        self._sb = self._ctx.__enter__()
        # Bound every WebDriver call. Without this, a page stuck loading (e.g. a
        # wedged Cloudflare interstitial) leaves the browser in a perpetual load
        # state and ALL commands block forever - the fetch thread never returns,
        # which hangs the whole cycle. With it, a stalled load raises instead, so
        # the retry/abandon logic below can actually run and free the browser.
        try:
            self._sb.driver.set_page_load_timeout(self._panel_wait_timeout)
        except Exception:
            log.warning("could not set page load timeout", exc_info=True)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)

    def fetch(self, job_id: str) -> JobDetails:
        url = _job_url(job_id)
        last_reason = "unknown"
        for attempt in range(1, _MAX_LOAD_ATTEMPTS + 1):
            # Open + solve captcha, then wait for the client panel to render. A
            # wedged Cloudflare interstitial (challenge clears but never hands off)
            # keeps the page loading forever, so the wait raises once the page-load
            # timeout set in __enter__ fires - we then reload, which usually
            # unsticks it. Re-opening the URL re-triggers the challenge, so each
            # retry re-solves the captcha via _open_and_solve_captcha.
            #
            # The panel anchor is member-since, present for every client, so we
            # never scrape before it renders - grabbing too early leaves client_*
            # fields None (a false "no prior hires").
            try:
                self._open_and_solve_captcha(job_id, url)
                self._sb.wait_for_element_present('[data-qa="client-contract-date"]', timeout=self._panel_wait_timeout)
            except Exception as e:
                detail = " ".join(str(e).split()) or "(no message)"
                last_reason = f"{type(e).__name__}: {detail}"[:200]
                log.warning("job %s did not load (attempt %d/%d): %s", job_id, attempt, _MAX_LOAD_ATTEMPTS, last_reason)
                continue

            html = self._sb.get_page_source()
            if os.environ.get("DEBUG_SCRAPE"):
                _dump_debug(html, job_id)
            return _extract(html, job_id, self._sb.get_current_url())

        # Exhausted all attempts - abandon so the pooled browser is freed for the
        # next job instead of blocking on a page that will not load.
        raise TimeoutError(f"job {job_id} did not load after {_MAX_LOAD_ATTEMPTS} attempts ({last_reason})")

    def _open_and_solve_captcha(self, job_id: str, url: str) -> None:
        """Open the job URL and click through the Cloudflare Turnstile. The click
        is serialized across pooled browsers by captcha_lock (it drives the real
        OS mouse, so only one can run at a time)."""
        self._sb.open(url)
        self._sb.sleep(2)
        if not self._captcha_lock.acquire(timeout=self._captcha_lock_timeout):
            raise TimeoutError(
                f"timed out waiting {self._captcha_lock_timeout}s for captcha_lock "
                f"(job {job_id}) - another instance's captcha click is likely hung"
            )
        try:
            self._sb.uc_gui_click_captcha()
        finally:
            self._captcha_lock.release()
        self._sb.sleep(2)


class BrowserPool:
    """Pool of `size` open BrowserFetcher instances sharing one captcha lock.

    Page-load + scrape overlap across instances; only the captcha click is
    serialized. Use as a context manager; check out an instance with acquire()
    and return it with release() (or use the `lease()` helper).
    """

    def __init__(
        self, size: int = 2, headless: bool = False, xvfb: bool = False,
        captcha_lock_timeout: float = 60.0, panel_wait_timeout: float = 20.0,
    ):
        self._size = max(1, size)
        self._headless = headless
        self._xvfb = xvfb
        self._captcha_lock_timeout = captcha_lock_timeout
        self._panel_wait_timeout = panel_wait_timeout
        self._captcha_lock = threading.Lock()
        self._available: queue.Queue[BrowserFetcher] = queue.Queue()
        self._all: list[BrowserFetcher] = []

    def __enter__(self):
        for _ in range(self._size):
            fetcher = BrowserFetcher(
                headless=self._headless,
                xvfb=self._xvfb,
                captcha_lock=self._captcha_lock,
                captcha_lock_timeout=self._captcha_lock_timeout,
                panel_wait_timeout=self._panel_wait_timeout,
            )
            fetcher.__enter__()
            self._all.append(fetcher)
            self._available.put(fetcher)
        return self

    def __exit__(self, *exc):
        for fetcher in self._all:
            try:
                fetcher.__exit__(*exc)
            except Exception:
                pass

    def acquire(self, timeout: float | None = None) -> BrowserFetcher:
        return self._available.get(timeout=timeout)

    def release(self, fetcher: BrowserFetcher) -> None:
        self._available.put(fetcher)

    def fetch(self, job_id: str) -> JobDetails:
        """Check out a browser, fetch, and return it to the pool."""
        fetcher = self.acquire()
        try:
            return fetcher.fetch(job_id)
        finally:
            self.release(fetcher)


if __name__ == "__main__":
    with BrowserFetcher() as bf:
        details = bf.fetch("2067753791591023938")
        print(details)
