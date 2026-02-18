#!/usr/bin/env python3
"""
Personalized Job Discovery Agent
=================================
Parses your resume, searches Adzuna for PM / Product Analytics roles,
scores each posting against your background using Claude, and prints a
ranked results table.

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt

Environment variables (see .env):
    ANTHROPIC_API_KEY
    ADZUNA_APP_ID
    ADZUNA_APP_KEY
"""

import csv
import hashlib
import json
import os
import sys
import time

import pdfplumber
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

RESUME_PATH = "SadhuPranati_Resume.pdf"

# Minimum Claude score (1-10) to include in results
MIN_SCORE = 6

# Maximum jobs to send to Claude (controls API cost; raise if you want more coverage)
MAX_JOBS_TO_SCORE = 40

# Adzuna country code — "us" for United States
ADZUNA_COUNTRY = "us"

# Jobs returned per Adzuna query (max 50)
RESULTS_PER_QUERY = 20

# Search queries sent to Adzuna — tuned for your background
SEARCH_QUERIES = [
    "Product Manager data analytics",
    "Product Analytics Manager",
    "Product Manager gaming",
    "Product Manager live service games",
    "Data driven Product Manager SQL BigQuery",
]

# ── Resume Parsing ────────────────────────────────────────────────────────────


def parse_resume(pdf_path: str) -> str:
    """Extract plain text from a PDF resume using pdfplumber."""
    if not os.path.exists(pdf_path):
        console.print(
            f"[red]Error:[/red] Resume not found at '[bold]{pdf_path}[/bold]'.\n"
            "Place 'SadhuPranati_Resume.pdf' in the same directory as main.py."
        )
        sys.exit(1)

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    full_text = "\n".join(pages).strip()

    if not full_text:
        console.print(
            "[red]Error:[/red] Could not extract text from the PDF. "
            "Ensure the file is not scanned/image-only."
        )
        sys.exit(1)

    return full_text


# ── Job Search (Adzuna API) ───────────────────────────────────────────────────


def search_jobs(query: str) -> list[dict]:
    """
    Search Adzuna for jobs matching *query*.

    Adzuna API docs: https://developer.adzuna.com/docs/search
    Returns a list of raw job dicts from the API.
    """
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")

    if not app_id or not app_key or app_id.startswith("your_"):
        console.print(
            "[red]Error:[/red] ADZUNA_APP_ID / ADZUNA_APP_KEY not configured.\n"
            "See the .env file for setup instructions."
        )
        sys.exit(1)

    url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "results_per_page": RESULTS_PER_QUERY,
        "sort_by": "relevance",
        "content-type": "application/json",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.HTTPError as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Adzuna returned HTTP {exc.response.status_code} "
            f"for query '{query}'. Skipping."
        )
        return []
    except requests.RequestException as exc:
        console.print(f"[yellow]Warning:[/yellow] Network error for '{query}': {exc}")
        return []


def deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate postings by Adzuna job ID (or a title+company hash)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for job in jobs:
        job_id = job.get("id") or hashlib.md5(
            (
                job.get("title", "")
                + job.get("company", {}).get("display_name", "")
            ).encode()
        ).hexdigest()
        if job_id not in seen:
            seen.add(job_id)
            unique.append(job)
    return unique


# ── Scoring (Anthropic / Claude) ──────────────────────────────────────────────

# The system instruction primes Claude as a strict, consistent recruiter.
SYSTEM_PROMPT = (
    "You are an expert technical recruiter evaluating candidate–job fit. "
    "You respond ONLY with valid JSON — no markdown, no prose outside the JSON object."
)

COVER_LETTER_SYSTEM_PROMPT = (
    "You are an expert career coach and professional writer specializing in tech and product roles. "
    "Write compelling, personalized cover letters that highlight specific alignment between "
    "the candidate's experience and the job requirements. Be direct, confident, and specific — "
    "no generic filler. Keep it under 350 words."
)

COVER_LETTER_TEMPLATE = """\
Write a personalized, compelling cover letter for this candidate applying to the job below.

Instructions:
- Open with a strong hook that references something specific about the company or role
- Highlight 2-3 concrete achievements from the resume that directly map to the job's requirements
- Weave in relevant skills (SQL, Python, BigQuery, analytics, gaming/live-service if applicable)
- Close with a confident call to action
- Tone: professional but warm, not robotic
- Do NOT use placeholder text like [Your Name] — write it as a complete, ready-to-send letter
- Sign off as: Pranati Sadhu

--- RESUME (truncated to first 4 000 chars) ---
{resume}

--- JOB POSTING ---
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

Cover letter:"""

# The user turn gives Claude the resume + job and asks for a structured score.
SCORE_TEMPLATE = """\
Evaluate how well the candidate's resume matches the job posting below.

IMPORTANT SCORING PRIORITIES (weight these heavily):
- Data-driven / analytics-focused PM roles
- Gaming or live-service product experience
- Roles requiring SQL, Python, or BigQuery
- Companies in interactive entertainment or consumer tech

Respond with ONLY this JSON (no other text):
{{"score": <integer 1-10>, "explanation": "<exactly 2 sentences>"}}

Scoring guide:
  9-10 → Exceptional fit: nearly all requirements met, highly relevant experience
  7-8  → Strong fit: most requirements met, background transfers well
  5-6  → Moderate fit: some relevant skills, noticeable gaps
  1-4  → Poor fit: significant mismatch in skills or domain

--- RESUME (truncated to first 4 000 chars) ---
{resume}

--- JOB POSTING ---
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

JSON response:"""


def generate_cover_letter(client: Anthropic, resume_text: str, job: dict) -> str:
    """
    Ask Claude to write a personalized cover letter for the given job.
    Returns the cover letter as a string, or an error message on failure.
    """
    title = job.get("title", "Unknown")
    company = job.get("company", {}).get("display_name", "Unknown")
    location = job.get("location", {}).get("display_name", "Unknown")
    description = (job.get("description") or "")[:3_000]

    prompt = COVER_LETTER_TEMPLATE.format(
        resume=resume_text[:4_000],
        title=title,
        company=company,
        location=location,
        description=description,
    )

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            system=COVER_LETTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001
        return f"Cover letter generation error: {exc}"


def score_job(client: Anthropic, resume_text: str, job: dict) -> tuple[int, str]:
    """
    Ask Claude to rate resume–job fit on a 1-10 scale.
    Returns (score, two_sentence_explanation).
    Returns (0, error_message) on failure so the job is filtered out.
    """
    title = job.get("title", "Unknown")
    company = job.get("company", {}).get("display_name", "Unknown")
    location = job.get("location", {}).get("display_name", "Unknown")
    # Trim description to keep prompt tokens manageable
    description = (job.get("description") or "")[:3_000]

    prompt = SCORE_TEMPLATE.format(
        resume=resume_text[:4_000],
        title=title,
        company=company,
        location=location,
        description=description,
    )

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Defensively strip any accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        data = json.loads(raw)
        score = max(1, min(10, int(data["score"])))
        explanation = str(data["explanation"])
        return score, explanation

    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as exc:
        return 0, f"Scoring parse error: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface unexpected API errors gracefully
        return 0, f"Scoring error: {exc}"


# ── Output ────────────────────────────────────────────────────────────────────

CSV_PATH = "job_matches.csv"


def save_csv(ranked_jobs: list[dict], include_cover_letters: bool = True) -> None:
    """Write ranked jobs to CSV. Always runs so results are never lost."""
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "title", "company", "location", "score",
                        "explanation", "cover_letter", "url"],
        )
        writer.writeheader()
        for rank, job in enumerate(ranked_jobs, start=1):
            writer.writerow({
                "rank": rank,
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "score": job["score"],
                "explanation": job["explanation"],
                "cover_letter": job.get("cover_letter", "") if include_cover_letters else "",
                "url": job["url"],
            })
    label = "with cover letters" if include_cover_letters else "scores only"
    console.print(f"[dim]Results saved to [bold]{CSV_PATH}[/bold] ({label})[/dim]")


def print_results(ranked_jobs: list[dict]) -> None:
    """Render a Rich table of the scored, filtered, ranked job matches."""
    if not ranked_jobs:
        return

    table = Table(
        title=f"[bold cyan]Top Job Matches[/bold cyan]  "
              f"[dim](score ≥ {MIN_SCORE}, ranked by fit)[/dim]",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        highlight=True,
    )
    table.add_column("#", style="dim", width=3, justify="right", no_wrap=True)
    table.add_column("Job Title", style="bold white", min_width=28)
    table.add_column("Company", style="cyan", min_width=18)
    table.add_column("Location", style="green", min_width=15)
    table.add_column("Score", justify="center", width=7, no_wrap=True)
    table.add_column("Why It Matches", min_width=45)
    table.add_column("Cover Letter", min_width=55)
    table.add_column("Apply", style="blue", min_width=10)

    for rank, job in enumerate(ranked_jobs, start=1):
        score = job["score"]
        if score >= 9:
            score_str = f"[bold green]{score}/10[/bold green]"
        elif score >= 7:
            score_str = f"[green]{score}/10[/green]"
        else:
            score_str = f"[yellow]{score}/10[/yellow]"

        table.add_row(
            str(rank),
            job["title"],
            job["company"],
            job["location"],
            score_str,
            job["explanation"],
            job.get("cover_letter", ""),
            job["url"],
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Showing {len(ranked_jobs)} qualified role(s) "
        f"out of {MAX_JOBS_TO_SCORE} scored.[/dim]"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def validate_env() -> None:
    """Abort early with a clear message if required env vars are missing."""
    missing = []
    for key in ("ANTHROPIC_API_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
        val = os.getenv(key, "")
        if not val or val.startswith("your_"):
            missing.append(key)
    if missing:
        console.print(
            "[red]Error:[/red] The following keys are not set in your [bold].env[/bold]:\n"
            + "\n".join(f"  • {k}" for k in missing)
        )
        sys.exit(1)


def main() -> None:
    validate_env()
    client = Anthropic()

    console.rule("[bold cyan]Personalized Job Discovery Agent[/bold cyan]")

    # ── Step 1: Parse resume ──────────────────────────────────────────────────
    with console.status("[bold]Parsing resume PDF...[/bold]"):
        resume_text = parse_resume(RESUME_PATH)
    console.print(f"[green]✓[/green] Resume parsed ({len(resume_text):,} characters)")

    # ── Step 2: Collect job postings ──────────────────────────────────────────
    all_jobs: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Searching Adzuna across {len(SEARCH_QUERIES)} queries...",
            total=len(SEARCH_QUERIES),
        )
        for query in SEARCH_QUERIES:
            results = search_jobs(query)
            all_jobs.extend(results)
            progress.advance(task)
            time.sleep(0.4)  # polite pause between API calls

    unique_jobs = deduplicate_jobs(all_jobs)
    console.print(
        f"[green]✓[/green] Found [bold]{len(unique_jobs)}[/bold] unique job postings"
    )

    if not unique_jobs:
        console.print(
            "[red]No jobs returned.[/red] Check your Adzuna credentials and network."
        )
        sys.exit(1)

    jobs_to_score = unique_jobs[:MAX_JOBS_TO_SCORE]
    if len(unique_jobs) > MAX_JOBS_TO_SCORE:
        console.print(
            f"[dim]Scoring {MAX_JOBS_TO_SCORE} of {len(unique_jobs)} jobs "
            f"(raise MAX_JOBS_TO_SCORE to increase coverage).[/dim]"
        )

    # ── Step 3: Score each job with Claude ────────────────────────────────────
    scored_jobs: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Scoring {len(jobs_to_score)} jobs with Claude...",
            total=len(jobs_to_score),
        )
        for job in jobs_to_score:
            score, explanation = score_job(client, resume_text, job)
            scored_jobs.append(
                {
                    "title": job.get("title", "Unknown"),
                    "company": job.get("company", {}).get("display_name", "Unknown"),
                    "location": job.get("location", {}).get("display_name", "Unknown"),
                    "score": score,
                    "explanation": explanation,
                    "url": job.get("redirect_url", "N/A"),
                }
            )
            progress.advance(task)

    # ── Step 4: Filter (score >= MIN_SCORE) and rank ─────────────────────────
    qualified = [j for j in scored_jobs if j["score"] >= MIN_SCORE]
    ranked = sorted(qualified, key=lambda j: j["score"], reverse=True)

    if not ranked:
        console.print(
            f"\n[yellow]No roles scored {MIN_SCORE}+ found.[/yellow] "
            "Try lowering MIN_SCORE or expanding SEARCH_QUERIES."
        )
        return

    # ── Step 5: Generate cover letters for qualified jobs ─────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating cover letters for {len(ranked)} qualified role(s)...",
            total=len(ranked),
        )
        for job in ranked:
            # Reconstruct the raw job dict fields needed by generate_cover_letter
            raw_job = {
                "title": job["title"],
                "company": {"display_name": job["company"]},
                "location": {"display_name": job["location"]},
                "description": next(
                    (j.get("description") for j in jobs_to_score
                     if j.get("title") == job["title"]
                     and j.get("company", {}).get("display_name") == job["company"]),
                    "",
                ),
            }
            job["cover_letter"] = generate_cover_letter(client, resume_text, raw_job)
            progress.advance(task)

    # ── Step 6: Display ───────────────────────────────────────────────────────
    print_results(ranked)

    # ── Step 7: Save outputs ───────────────────────────────────────────────────
    import sys as _sys
    cwd = os.getcwd()
    console.print(f"[dim]Saving files to: [bold]{cwd}[/bold][/dim]")

    try:
        save_csv(ranked, include_cover_letters=True)
    except Exception as exc:
        console.print(f"[red]ERROR saving CSV:[/red] {exc}", file=_sys.stderr)

    cover_letters_dir = "cover_letters"
    try:
        os.makedirs(cover_letters_dir, exist_ok=True)
        saved = 0
        for rank, job in enumerate(ranked, start=1):
            letter = job.get("cover_letter", "")
            if letter and not letter.startswith("Cover letter generation error"):
                safe_company = "".join(c if c.isalnum() or c in " _-" else "" for c in job["company"])
                safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in job["title"])
                filename = f"{rank:02d}_{safe_company}_{safe_title}.txt".replace(" ", "_")[:80]
                filepath = os.path.join(cover_letters_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Position: {job['title']}\n")
                    f.write(f"Company:  {job['company']}\n")
                    f.write(f"Location: {job['location']}\n")
                    f.write(f"Score:    {job['score']}/10\n")
                    f.write(f"Apply:    {job['url']}\n")
                    f.write("\n" + "─" * 60 + "\n\n")
                    f.write(letter)
                saved += 1
        console.print(f"[dim]{saved} cover letter(s) saved to [bold]{cover_letters_dir}/[/bold][/dim]")
    except Exception as exc:
        console.print(f"[red]ERROR saving cover letters:[/red] {exc}", file=_sys.stderr)


if __name__ == "__main__":
    main()
