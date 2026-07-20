"""LLM scope filter - runs on the cheap fast-pass data (title/description/skills)
BEFORE the expensive browser fetch, so only genuinely relevant jobs get scraped.

Replaces the old regex prefilter: a keyword can match incidentally (e.g. a job that
mentions "react" in passing but is really a WordPress gig). The LLM judges whether the
job is actually within the freelancer's services, not just a surface keyword hit.

One gpt-4.1-mini call per job; the pipeline runs these concurrently.
"""

import json
from dataclasses import dataclass, field

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from job_fetcher import JobSummary

MODEL = "gpt-4.1-mini"

_SYSTEM_PROMPT = """You are screening Upwork job posts for a freelancer. Your job is to decide \
whether the PRIMARY technical deliverable of the job matches the freelancer's core skill set - \
NOT whether they have experience with every tool, platform, or niche domain mentioned.

Judgement rules:
- PASS if the dominant/core tech requirement (e.g. build an AI agent, build a RAG chatbot, \
build a FastAPI backend, build a Next.js app) maps to the freelancer's main stack, even if \
the job also mentions peripheral tools, third-party platforms, or domain-specific experience \
the freelancer hasn't listed (e.g. GHL, Salesforce, specific CRMs, specific industry knowledge).
- PASS if the freelancer has the relevant engineering skill even without niche domain experience \
(e.g. "build a RAG chatbot for a law firm" = pass if they know RAG, not a pass only if they \
know law; "build an ecommerce site in Next.js" = pass if they know Next.js, not only if they \
have ecommerce-specific experience).
- REJECT only if the core/primary deliverable is genuinely outside the freelancer's stack \
(e.g. the job is mainly a WordPress site, a mobile app in Swift, a Salesforce admin role, \
video editing, or is non-technical/spam).
- When in doubt, PASS - downstream scoring and a browser check will filter further. \
Favour recall over precision."""

# Structured verdict - guarantees a parseable pass/fail without prompt-format fragility.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "matched_areas": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["relevant", "matched_areas", "reason"],
    "additionalProperties": False,
}


@dataclass
class ScopeVerdict:
    relevant: bool
    matched_areas: list[str] = field(default_factory=list)
    reason: str = ""


def _scope_context(config: dict) -> str:
    """Human-readable skill areas + profile summary, used as relevance context."""
    lines = []
    for s in config.get("scope", []):
        kws = ", ".join(s.get("keywords", []))
        lines.append(f"- {s['name']}: {kws}")
    scope_block = "\n".join(lines)

    p = config.get("freelancer_profile", {})
    stack = p.get("tech_stack", {})
    all_tech = stack.get("ai_ml", []) + stack.get("full_stack", [])
    return (
        f"Freelancer title: {p.get('title', '')}\n"
        f"Service areas:\n{scope_block}\n"
        f"Tech stack: {', '.join(all_tech)}"
    )


def scope_filter_job(summary: JobSummary, config: dict, client: OpenAI | None = None) -> ScopeVerdict:
    client = client or OpenAI()  # reads OPENAI_API_KEY

    user_prompt = (
        f"{_scope_context(config)}\n\n"
        f"--- JOB POST ---\n"
        f"Title: {summary.title}\n"
        f"Skills tagged: {', '.join(summary.skills) or 'none'}\n"
        f"Matched search terms: {', '.join(summary.matched_keywords) or 'none'}\n"
        f"Description:\n{summary.description}\n\n"
        f"Does the PRIMARY technical deliverable of this job match the freelancer's core skills? "
        f"Ignore peripheral tools, niche domain knowledge, or add-on platform experience "
        f"(e.g. GHL, specific CRMs, industry-specific apps) that aren't the main technical challenge. The matched keywords are just surface hits; judge the actual deliverable. Like if the Job is relevant to the freelancer's skills in broad way, PASS it even if the matched keywords are not high or the freelancer doesn't have experience with every tool/platform/domain mentioned. If the job is clearly outside the freelancer's scope + Highly irrelevant tech stack (for example OS programming but in C++/Java etc... which does not align with the freelancer's skills + scope + tech stack), REJECT it. When in doubt, PASS. Give a short reason for your decision. The core thing is that if there is Every tool/skill/domain is not required to be present in freelancer's profile"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scope_verdict", "strict": True, "schema": _VERDICT_SCHEMA},
        },
    )

    data = json.loads(response.choices[0].message.content)
    return ScopeVerdict(
        relevant=bool(data["relevant"]),
        matched_areas=list(data.get("matched_areas", [])),
        reason=data.get("reason", ""),
    )


if __name__ == "__main__":
    import yaml

    from job_fetcher import JobFetcher

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    jobs = JobFetcher().fetch_recent(max_pages=1)
    for job in jobs[:5]:
        verdict = scope_filter_job(job, cfg)
        print(f"[{'PASS' if verdict.relevant else 'REJECT'}] {job.title} -> {verdict.reason}")
