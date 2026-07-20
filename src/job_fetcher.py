"""Fast-pass job fetcher: pulls title/description/skills/jobType/tier/publishTime
via Upwork's GraphQL API using a visitor (unauthenticated) token.

Confirmed via live testing that the visitor token is blocked from ciphertext/link/url,
client info, and applicant counts - those are fetched separately in browser_fetcher.py.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from token_manager import TokenManager

GRAPHQL_URL = "https://www.upwork.com/api/graphql/v1"
PAGE_SIZE = 50

# Hard backstop on pages walked per keyword. The publishTime window is the real
# paging bound (see _fetch_new_for_keyword); this only guards a runaway crawl if
# timestamps are missing or the whole window is somehow full.
MAX_PAGES_HARD_CAP = 10


def _parse_publish_time(ts: str | None) -> datetime | None:
    """Parse Upwork's ISO-8601 publishTime to an aware UTC datetime, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Treat a naive timestamp as UTC so comparisons never raise.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

_QUERY = """
query VisitorJobSearch($requestVariables: VisitorJobSearchV1Request!) {
  search {
    universalSearchNuxt {
      visitorJobSearchV1(request: $requestVariables) {
        paging { total offset count }
        results {
          id
          title
          description
          ontologySkills { prefLabel }
          jobTile {
            job {
              jobType
              contractorTier
              publishTime
            }
          }
        }
      }
    }
  }
}
"""


@dataclass
class JobSummary:
    id: str
    title: str
    description: str
    skills: list[str] = field(default_factory=list)
    job_type: str | None = None
    contractor_tier: str | None = None
    publish_time: str | None = None
    matched_keywords: list[str] = field(default_factory=list)  # search terms this job came back for


class JobFetcher:
    def __init__(self, token_manager: TokenManager | None = None):
        self._tm = token_manager or TokenManager()

    def _post(self, variables: dict) -> dict:
        token = self._tm.get_token()
        resp = self._tm.session.post(
            GRAPHQL_URL,
            json={"query": _QUERY, "variables": {"requestVariables": variables}},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            impersonate="chrome",
            timeout=30,
        )
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"Upwork GraphQL error: {data['errors']}")
        return data["data"]["search"]["universalSearchNuxt"]["visitorJobSearchV1"]

    def fetch_page(self, offset: int = 0, count: int = PAGE_SIZE, user_query: str = "") -> tuple[list[JobSummary], int]:
        result = self._post(
            {
                "userQuery": user_query,
                "sort": "recency",
                "paging": {"offset": offset, "count": count},
            }
        )
        jobs = []
        for r in result["results"]:
            job_tile = (r.get("jobTile") or {}).get("job") or {}
            jobs.append(
                JobSummary(
                    id=r["id"],
                    title=r["title"],
                    description=r.get("description") or "",
                    skills=[s["prefLabel"] for s in (r.get("ontologySkills") or [])],
                    job_type=job_tile.get("jobType"),
                    contractor_tier=job_tile.get("contractorTier"),
                    publish_time=job_tile.get("publishTime"),
                )
            )
        return jobs, result["paging"]["total"]

    def fetch_recent(self, max_pages: int = 3, user_query: str = "") -> list[JobSummary]:
        jobs = []
        for page in range(max_pages):
            page_jobs, _total = self.fetch_page(offset=page * PAGE_SIZE, count=PAGE_SIZE, user_query=user_query)
            if not page_jobs:
                break
            jobs.extend(page_jobs)
        return jobs

    def _fetch_new_for_keyword(self, keyword: str, store, cutoff: datetime) -> list[JobSummary]:
        """Walk recency-sorted results newest->oldest, collecting jobs posted at or
        after `cutoff` that aren't already in the store.

        The publishTime window is the paging bound: because results are recency
        sorted, the first job older than `cutoff` means every later job is older
        too, so we stop. `store.is_seen` is used only to dedup the deliberate
        window overlap (so a job re-fetched on the next run isn't emitted twice) -
        it is no longer the stop condition. MAX_PAGES_HARD_CAP guards a runaway.

        Jobs with an unparseable/missing publishTime are kept (fail open) and do
        not trigger the stop, so a timestamp glitch can't silently drop a job.
        """
        new_jobs: list[JobSummary] = []
        for page in range(MAX_PAGES_HARD_CAP):
            page_jobs, _total = self.fetch_page(offset=page * PAGE_SIZE, count=PAGE_SIZE, user_query=keyword)
            if not page_jobs:
                break
            reached_window_edge = False
            for j in page_jobs:
                published = _parse_publish_time(j.publish_time)
                if published is not None and published < cutoff:
                    reached_window_edge = True
                    break
                if not store.is_seen(j.id):
                    j.matched_keywords = [keyword]
                    new_jobs.append(j)
            if reached_window_edge:
                break
        return new_jobs

    def fetch_new(self, keywords: list[str], store, lookback_minutes: int = 30, max_workers: int = 8) -> list[JobSummary]:
        """Concurrent multi-keyword incremental fetch, bounded by a time window.

        Runs one recency-sorted search per keyword in parallel and returns jobs
        posted within the last `lookback_minutes` that aren't already in `store`,
        deduped across keywords (a job matching several keywords appears once with
        all matched_keywords merged).

        Set `lookback_minutes` wider than the poll interval: the overlap lets a
        late or failed run self-heal on the next cycle, and `store`'s dedup means
        the re-fetched overlap is never emitted twice.
        """
        # Pre-warm the visitor token so the concurrent searches share one cached
        # token instead of racing to refresh it.
        self._tm.get_token()

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        merged: dict[str, JobSummary] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(keywords)))) as ex:
            futures = {
                ex.submit(self._fetch_new_for_keyword, kw, store, cutoff): kw
                for kw in keywords
            }
            for fut in as_completed(futures):
                for job in fut.result():
                    existing = merged.get(job.id)
                    if existing is None:
                        merged[job.id] = job
                    else:
                        # same job from another keyword search - merge matched keywords
                        for kw in job.matched_keywords:
                            if kw not in existing.matched_keywords:
                                existing.matched_keywords.append(kw)
        return list(merged.values())


if __name__ == "__main__":
    fetcher = JobFetcher()
    jobs = fetcher.fetch_recent(max_pages=1)
    print(f"fetched {len(jobs)} jobs")
    for j in jobs[:5]:
        print(f"- [{j.job_type}/{j.contractor_tier}] {j.title} | skills={j.skills[:3]}")
