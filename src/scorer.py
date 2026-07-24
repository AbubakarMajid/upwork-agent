"""Combines fast-pass (job_fetcher) + slow-pass (browser_fetcher) data and scores
a job against config.yaml's rules.

Acceptance is a hard gate (all must pass, in order - first failure rejects):
  - job was posted within the max age window (default 10h - old jobs go cold fast)
  - budget meets the stated minimum ($/hr or fixed) IF one is published; a job
    with no readable rate is not rejected (guest view hides many rates and the
    client's average rate isn't available unauthenticated - can't be verified)
  - proposal count does not exceed the max (reject if already > 15 proposals)
  - client isn't already interviewing anyone
  - client hasn't already invited other freelancers
  - this posting hasn't already hired anyone (job effectively filled = reject)
  - client has at least one prior hire (no hire history = reject)
  - client isn't located in a rejected country

Everything else (client spend/hires, budget size) is pure scoring - it ranks
accepted jobs, it never rejects them.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from browser_fetcher import JobDetails
from job_fetcher import JobSummary

_PROPOSAL_TIER_MAX = {
    "Less than 5": 5,
    "5 to 10": 10,
    "10 to 15": 15,
    "15 to 20": 20,
    "20 to 50": 50,
    "50+": 999,
}


@dataclass
class ScoreResult:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_scope: list[str] = field(default_factory=list)


def _parse_money(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"\$([\d,.]+)\s*([Kk]?)", text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    if m.group(2):
        value *= 1000
    return value


def _parse_hires(text: str | None) -> int | None:
    """Parse a hire count from Upwork's stat text. Handles the formats established
    clients show: "12 hires", "500+ hires", "1K+ hires", "1,234 hires"."""
    if not text:
        return None
    m = re.search(r"([\d,.]+)\s*([KkMm]?)\+?\s*hires?", text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    value *= {"k": 1_000, "m": 1_000_000}.get(m.group(2).lower(), 1)
    return int(value)


def _hours_since(iso_timestamp: str | None) -> float | None:
    if not iso_timestamp:
        return None
    try:
        posted = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - posted).total_seconds() / 3600


def _parse_count(text: str | None) -> int:
    """Parse a leading count out of a string like "2" or "1 interviewing".

    Upwork sometimes omits the row entirely when the count is zero, so a
    missing field means 0, not "unknown".
    """
    if not text:
        return 0
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def score_job(
    summary: JobSummary,
    details: JobDetails,
    matched_scope: list[str],
    config: dict,
) -> ScoreResult:
    rules = config["scoring"]

    if not matched_scope:
        return ScoreResult(passed=False, score=0, reasons=["no scope match"])

    # --- Hard acceptance gates ---
    max_age_hours = rules.get("max_job_age_hours")
    if max_age_hours is not None:
        age_hours = _hours_since(summary.publish_time)
        if age_hours is None:
            return ScoreResult(passed=False, score=0, reasons=["publish time not found - could not verify age"])
        if age_hours > max_age_hours:
            return ScoreResult(
                passed=False, score=0,
                reasons=[f"job is {age_hours:.1f}h old, older than the {max_age_hours}h limit"],
            )

    hourly_min = _parse_money(details.hourly_min)
    hourly_max = _parse_money(details.hourly_max)
    fixed_price = _parse_money(details.fixed_price)

    # For an hourly range ("$7-$30") the client's ceiling is what matters: if the
    # top of the range clears our floor they can pay our rate, so we bid within the
    # range. Gate on the ceiling (hourly_max), falling back to hourly_min for a
    # single-value posting.
    hourly_ceiling = hourly_max if hourly_max is not None else hourly_min

    # Budget gate fails open: reject only when a rate/budget is actually present
    # AND below the minimum. When it can't be read (rate hidden in guest view, or
    # an unknown budget type) we let the job through - the client's average rate
    # isn't available unauthenticated, so an absent rate is unverifiable, not a
    # rejection. The real floors below still apply to jobs that publish a rate.
    if details.budget_type == "Hourly" and hourly_ceiling is not None:
        if hourly_ceiling < rules["min_pay_rate_usd_per_hr"]:
            return ScoreResult(
                passed=False, score=0,
                reasons=[f"hourly rate ${hourly_ceiling:g} below minimum ${rules['min_pay_rate_usd_per_hr']}/hr"],
            )
    elif details.budget_type in ("Fixed-price", "Fixed price") and fixed_price is not None:
        if fixed_price < rules["min_fixed_price_usd"]:
            return ScoreResult(
                passed=False, score=0,
                reasons=[f"fixed price ${fixed_price:g} below minimum ${rules['min_fixed_price_usd']}"],
            )

    proposal_tier_max = _PROPOSAL_TIER_MAX.get(details.proposals)
    max_proposal_count = rules["max_proposal_count"]
    if proposal_tier_max is not None and proposal_tier_max > max_proposal_count:
        return ScoreResult(
            passed=False, score=0,
            reasons=[f"too many proposals ({details.proposals}, max is {max_proposal_count})"],
        )

    interviewing = _parse_count(details.interviewing)
    if rules.get("reject_if_interviewing", True) and interviewing > 0:
        return ScoreResult(passed=False, score=0, reasons=[f"client already interviewing ({details.interviewing})"])

    invites = _parse_count(details.invites_sent)
    if rules.get("reject_if_freelancers_invited", True) and invites > 0:
        return ScoreResult(passed=False, score=0, reasons=[f"freelancers already invited ({details.invites_sent})"])

    # Hires on this specific posting: if the client already hired anyone for this
    # job, it's effectively filled - reject. Absent row means 0 (see _extract).
    job_hires = _parse_count(details.job_hires)
    if rules.get("reject_if_job_already_hired", True) and job_hires > 0:
        return ScoreResult(passed=False, score=0, reasons=[f"job already has {job_hires} hire(s) ({details.job_hires})"])

    hires = _parse_hires(details.client_hires)
    if rules.get("reject_if_no_hires", True) and not hires:
        return ScoreResult(
            passed=False, score=0,
            reasons=[f"client has no prior hires (raw client_hires={details.client_hires!r})"],
        )

    reject_countries = [c.lower() for c in rules.get("reject_countries", [])]
    if reject_countries and details.client_location:
        location = details.client_location.lower()
        if any(country in location for country in reject_countries):
            return ScoreResult(passed=False, score=0, reasons=[f"client location restricted ({details.client_location})"])

    # --- Scoring (accepted jobs only) ---
    reasons: list[str] = []
    score = 10 * len(matched_scope)

    # Budget may be absent (gate fails open above), so only score it when present.
    # Score on the ceiling so a range like $7-$30 is judged on its $30 top, not $7.
    if details.budget_type == "Hourly" and hourly_ceiling is not None:
        score += min(hourly_ceiling, 50)
    elif fixed_price is not None:
        score += min(fixed_price / 20, 50)

    if proposal_tier_max is not None:
        score += max(0, max_proposal_count - proposal_tier_max)
        reasons.append(f"proposals: {details.proposals}")

    spend = _parse_money(details.client_total_spent)
    if spend:
        score += min(spend / 1000, 20)
        reasons.append(f"client spend: {details.client_total_spent}")

    score += min(hires or 0, 10)
    reasons.append(f"client hires: {hires}")

    if details.country_restricted_note:
        reasons.append(f"NOTE - eligibility restriction found: {details.country_restricted_note} (verify manually)")

    reasons.append(f"matched scope: {', '.join(matched_scope)}")
    return ScoreResult(passed=True, score=round(score, 1), reasons=reasons, matched_scope=matched_scope)
