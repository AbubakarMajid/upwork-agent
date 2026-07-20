"""Drafts a tailored Upwork proposal via OpenAI (GPT-4.1-mini), using config.yaml's freelancer
profile and the two writing-style modes (general role vs. specific project).

Pricing is computed deterministically in Python (not left to the model) and handed to
the prompt as an instruction - this is more reliable than asking the LLM to do the
budget math itself.
"""

import re

from openai import OpenAI

from browser_fetcher import JobDetails
from job_fetcher import JobSummary

MODEL = "gpt-4.1-mini"

_SYSTEM_PROMPT = """You are drafting Upwork job proposals on behalf of Abubakar Majid, a \
freelance AI/full-stack developer. Write in first person as the freelancer. Be concrete and \
specific - avoid generic filler ("I am excited to apply...", "I would love to help..."). \
The client is usually non-technical, so explain technical work in plain, practical language \
rather than jargon. Do not use markdown headers, bullet symbols, or bold formatting - this is \
plain text for a chat message. Follow the section structure and word-count limits given exactly, \
and skip any section marked optional when instructed to. End with exactly this sign-off on its \
own line: "Regards,\\nAbubakar Majid". Output only the proposal text, nothing else."""

_GENERAL_TEMPLATE = """Job title: {title}
Job description: {description}

This is a general role-type job posting (e.g. "Senior AI Engineer", "Next.js Developer" - an \
ongoing role rather than one specific deliverable). Write the proposal in this exact structure:

1. Brief intro (20-25 words max, first person, no filler).
2. Relevant projects/apps I've built - emphasize the live, running PRODUCTION apps from my \
portfolio that best match this role (not side projects). Pick the 1-2 most relevant.
3. Availability: I'm available full-time.
4. Rate: {rate_guidance}
5. Why I'm the best fit: I can handle this full-stack myself - the core skill they need plus \
related development work - so they don't need to hire multiple freelancers.
6. One brief closing line looking forward to connecting.
7. Sign-off (see system instructions).

Freelancer profile:
{profile}

---
Reference proposal (DO NOT copy this - use it only to match the tone, confidence, and depth of \
detail. Adapt everything to the specific job above):
{example}
---
"""

_PROJECT_TEMPLATE = """Job title: {title}
Job description: {description}

This is a specific project/problem-type job posting (a defined deliverable or problem to solve). \
Write the proposal in this exact structure:

1. Brief intro (20-25 words max, first person, no filler).
2. Solution approach: describe WHAT I'd do and HOW, in practical non-technical language - most \
clients here aren't technical. Name the tech stack and give a high-level implementation plan. \
Keep this concise (a few sentences, not exhaustive) - mention a rough timeline only if the job \
post asks for one.
3. Relevant projects/apps I've built - emphasize the live, running PRODUCTION apps from my \
portfolio that best match this work (not side projects). Pick the 1-2 most relevant.
4. Availability: I'm available full-time.
5. Rate: {rate_guidance}
6. Why I'm the best fit (OPTIONAL - skip this section entirely if the job post explicitly asks \
for only a specific narrow deliverable, e.g. "just send me X"): I can handle this full-stack \
myself - the core need plus related development work.
7. One brief closing line.
8. Sign-off (see system instructions).

Freelancer profile:
{profile}

---
Reference proposal (DO NOT copy this - use it only to match the tone, confidence, solution \
structure, and depth of detail. Adapt everything to the specific job above):
{example}
---
"""

_ROLE_KEYWORDS = ["developer", "engineer", "specialist", "expert", "consultant", "manager", "designer"]


def _parse_money(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"\$([\d,.]+)", text)
    return float(m.group(1).replace(",", "")) if m else None


def _rate_guidance(details: JobDetails, max_rate: float) -> str:
    if details.budget_type == "Hourly":
        client_rate = _parse_money(details.hourly_min)
        if client_rate is None:
            return f"Quote a competitive hourly rate up to my ${max_rate:g}/hr ceiling."
        if client_rate <= max_rate:
            return f"Quote ${client_rate:g}/hr, matching the client's posted rate exactly."
        low = max(max_rate - 5, max_rate * 0.75)
        return (
            f"The client's posted rate (${client_rate:g}/hr) is above my ${max_rate:g}/hr ceiling - "
            f"quote competitively between ${low:g}-${max_rate:g}/hr, not the client's full rate."
        )
    if details.budget_type == "Fixed-price":
        fixed = _parse_money(details.fixed_price)
        budget_note = f" (their stated budget is ${fixed:g})" if fixed else ""
        return (
            f"This is a fixed-price job{budget_note}. Propose a price within the client's stated "
            f"budget - at or under it if their budget is reasonable, competitively rather than "
            f"over it if their budget looks tight."
        )
    return f"Quote a competitive rate up to my ${max_rate:g}/hr ceiling."


def _profile_text(config: dict) -> str:
    p = config["freelancer_profile"]
    portfolio_lines = "\n".join(f"- {item['name']}: {item['summary']} (tech: {', '.join(item['tech'])})" for item in p["portfolio"])
    return (
        f"Title: {p['title']}\n"
        f"Badges: {', '.join(p['badges'])}\n"
        f"Experience: {p['years_experience']}+ years, {p['projects_delivered']}+ projects delivered\n"
        f"Portfolio (all live production apps):\n{portfolio_lines}\n"
        f"Tech stack: {', '.join(p['tech_stack']['ai_ml'] + p['tech_stack']['full_stack'])}"
    )


def _is_general_role_job(title: str) -> bool:
    lowered = title.lower()
    return any(kw in lowered for kw in _ROLE_KEYWORDS) and len(title.split()) <= 6


def draft_proposal(summary: JobSummary, details: JobDetails, config: dict, client: OpenAI | None = None) -> str:
    client = client or OpenAI()  # reads OPENAI_API_KEY
    profile = _profile_text(config)
    max_rate = config["freelancer_profile"]["hourly_rate_usd"]
    rate_guidance = _rate_guidance(details, max_rate)

    is_general = _is_general_role_job(summary.title)
    template = _GENERAL_TEMPLATE if is_general else _PROJECT_TEMPLATE
    examples = config.get("proposal_examples", {})
    example = examples.get("general" if is_general else "project", "").strip()

    user_prompt = template.format(
        title=summary.title,
        description=summary.description,
        profile=profile,
        rate_guidance=rate_guidance,
        example=example,
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=900,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import yaml

    from browser_fetcher import BrowserFetcher
    from job_fetcher import JobFetcher
    from scope_filter import scope_filter_job

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    jobs = JobFetcher().fetch_recent(max_pages=1)
    job = next(j for j in jobs if scope_filter_job(j, cfg).relevant)

    with BrowserFetcher() as bf:
        details = bf.fetch(job.id)

    print(f"Job: {job.title}\n")
    print(draft_proposal(job, details, cfg))
