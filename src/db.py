"""SQLite store - tracks which jobs have been seen/evaluated and which were notified.

`status` lets the pipeline avoid re-work across cycles:
  - "seen"     : fetched and about to be (or being) evaluated
  - "rejected" : evaluated, did not pass (LLM scope filter or scoring)
  - "notified" : passed and sent to Slack

Incremental fetch pages back by a publishTime window (not by is_seen); is_seen()
dedups that window's overlap so the LLM scope filter never re-runs on a job
evaluated in a previous cycle.
"""

import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "resources" / "jobs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_jobs (
    job_id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'seen',
    publish_time TEXT,
    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class JobStore:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the store is shared across the pipeline's worker
        # threads; a lock serializes writes (SQLite connections aren't thread-safe).
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def is_seen(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT 1 FROM seen_jobs WHERE job_id = ?", (job_id,))
            return cur.fetchone() is not None

    # kept for backwards compatibility with callers that only care about notification
    def already_notified(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM seen_jobs WHERE job_id = ? AND status = 'notified'", (job_id,)
            )
            return cur.fetchone() is not None

    def mark_seen(self, job_id: str, title: str | None = None, publish_time: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_jobs (job_id, title, status, publish_time) "
                "VALUES (?, ?, 'seen', ?)",
                (job_id, title, publish_time),
            )
            self._conn.commit()

    def _set_status(self, job_id: str, status: str, title: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO seen_jobs (job_id, title, status, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(job_id) DO UPDATE SET status = excluded.status, "
                "title = COALESCE(excluded.title, seen_jobs.title), "
                "updated_at = CURRENT_TIMESTAMP",
                (job_id, title, status),
            )
            self._conn.commit()

    def mark_rejected(self, job_id: str, title: str | None = None) -> None:
        self._set_status(job_id, "rejected", title)

    def mark_notified(self, job_id: str, title: str | None = None) -> None:
        self._set_status(job_id, "notified", title)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class NullJobStore:
    """No-op store for dev/testing. is_seen() is always False and every mark_* is
    a no-op, so nothing is persisted and every run re-processes the same jobs.

    Upwork serves a given job to the anonymous fetcher only once, so with the real
    store a job can't be re-run for debugging once it's marked seen; this store
    removes that barrier. Selected via the PERSIST_JOBS env var in main.py.
    """

    def is_seen(self, job_id: str) -> bool:
        return False

    def already_notified(self, job_id: str) -> bool:
        return False

    def mark_seen(self, job_id: str, title: str | None = None, publish_time: str | None = None) -> None:
        pass

    def mark_rejected(self, job_id: str, title: str | None = None) -> None:
        pass

    def mark_notified(self, job_id: str, title: str | None = None) -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == "__main__":
    store = JobStore()
    print("is_seen('test123'):", store.is_seen("test123"))
    store.mark_seen("test123", "Test Job", publish_time="2026-06-19T00:00:00Z")
    print("is_seen('test123') after mark:", store.is_seen("test123"))
    store.mark_notified("test123", "Test Job")
    print("already_notified('test123'):", store.already_notified("test123"))
