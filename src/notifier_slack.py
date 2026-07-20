"""Sends formatted job alerts to Slack via an Incoming Webhook."""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from browser_fetcher import JobDetails
from job_fetcher import JobSummary
from scorer import ScoreResult

load_dotenv()

# Slack webhook payloads are capped at 40,000 chars; stay well under that.
_MAX_TEXT_LEN = 39000
_MAX_DESCRIPTION_LEN = 1500


def _time_since(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return "unknown"
    try:
        posted = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    delta = datetime.now(timezone.utc) - posted
    seconds = delta.total_seconds()
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _budget_display(details: JobDetails) -> str:
    if details.budget_type == "Hourly":
        if details.hourly_min and details.hourly_max and details.hourly_min != details.hourly_max:
            return f"{details.hourly_min}-{details.hourly_max}/hr"
        if details.hourly_min:
            return f"{details.hourly_min}/hr"
        return "Hourly - rate n/a"
    if details.budget_type == "Fixed-price":
        return details.fixed_price or "Fixed price - amount n/a"
    return "n/a"


def _format_message(summary: JobSummary, details: JobDetails, result: ScoreResult, proposal: str) -> str:
    description = summary.description.strip()
    if len(description) > _MAX_DESCRIPTION_LEN:
        description = description[:_MAX_DESCRIPTION_LEN].rstrip() + "..."

    return (
        f"*{summary.title}*\n"
        f"Score: {result.score} | Posted: {_time_since(summary.publish_time)}\n"
        f"Budget: {details.budget_type or 'n/a'} - {_budget_display(details)}\n"
        f"Proposals: {details.proposals or 'n/a'}\n"
        f"Client spend: {details.client_total_spent or 'n/a'} | "
        f"Hires: {details.client_hires or 'n/a'} | "
        f"Jobs posted: {details.client_jobs_posted or 'n/a'}\n\n"
        f"*Description:*\n{description}\n\n"
        f"*Apply:* <{details.url}>\n\n"
        f"*Draft proposal:*\n{proposal}"
    )


class SlackNotifier:
    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.environ["SLACK_WEBHOOK_URL"]

    def send_job_alert(self, summary: JobSummary, details: JobDetails, result: ScoreResult, proposal: str) -> None:
        text = _format_message(summary, details, result, proposal)[:_MAX_TEXT_LEN]
        resp = requests.post(
            self._webhook_url,
            json={"text": text, "mrkdwn": True},
            timeout=15,
        )
        resp.raise_for_status()

    def send_text(self, text: str) -> None:
        resp = requests.post(
            self._webhook_url,
            json={"text": text},
            timeout=15,
        )
        resp.raise_for_status()


if __name__ == "__main__":
    notifier = SlackNotifier()
    notifier.send_text("Upwork agent test message - if you see this, the webhook is wired up correctly.")
    print("sent")
