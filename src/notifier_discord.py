"""Sends formatted job alerts to Discord via an Incoming Webhook.

Mirrors the SlackNotifier interface (send_job_alert / send_text) but renders
alerts as rich embeds: a clickable title, a colored sidebar, and inline fields
for the key metrics.
"""

import os
import json
from datetime import datetime, timezone
from urllib import error, request

from dotenv import load_dotenv

from browser_fetcher import JobDetails
from job_fetcher import JobSummary
from scorer import ScoreResult

load_dotenv()

# Discord embed limits: description <= 4096, field value <= 1024, total <= 6000.
# Stay comfortably under so title/fields never push the payload over the edge.
_MAX_DESCRIPTION_LEN = 1400
_MAX_PROPOSAL_LEN = 1800
_GREEN = 0x2ECC71  # sidebar color for accepted jobs


def _post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Discord/Cloudflare rejects urllib's default UA with a 403; send a real one.
            "User-Agent": "upwork-agent (https://github.com, 1.0)",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Discord webhook failed with status {resp.status}")
    except error.HTTPError as exc:
        raise RuntimeError(f"Discord webhook failed with status {exc.code}") from exc


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


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _build_embed(summary: JobSummary, details: JobDetails, result: ScoreResult, proposal: str) -> dict:
    description = _truncate(summary.description, _MAX_DESCRIPTION_LEN)
    proposal_text = _truncate(proposal, _MAX_PROPOSAL_LEN)

    body = f"**Description**\n{description}\n\n**Draft proposal**\n{proposal_text}"

    return {
        "title": _truncate(summary.title, 250),
        "url": details.url,
        "color": _GREEN,
        "description": body,
        "fields": [
            {"name": "Score", "value": str(result.score), "inline": True},
            {"name": "Posted", "value": _time_since(summary.publish_time), "inline": True},
            {"name": "Budget", "value": f"{details.budget_type or 'n/a'} - {_budget_display(details)}", "inline": True},
            {"name": "Proposals", "value": details.proposals or "n/a", "inline": True},
            {"name": "Client spend", "value": details.client_total_spent or "n/a", "inline": True},
            {"name": "Hires", "value": details.client_hires or "n/a", "inline": True},
        ],
    }


class DiscordNotifier:
    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.environ["DISCORD_WEBHOOK_URL"]

    def send_job_alert(self, summary: JobSummary, details: JobDetails, result: ScoreResult, proposal: str) -> None:
        embed = _build_embed(summary, details, result, proposal)
        _post_json(self._webhook_url, {"embeds": [embed]})

    def send_text(self, text: str) -> None:
        # Discord message content is capped at 2000 chars.
        _post_json(self._webhook_url, {"content": text[:2000]})


if __name__ == "__main__":
    notifier = DiscordNotifier()
    notifier.send_text("Upwork agent test message - if you see this, the Discord webhook is wired up correctly.")
    print("sent")
