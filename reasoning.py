"""
reasoning.py
------------
Generates the reasoning column for each top-100 candidate.

Rules from submission spec Stage 4 review:
- Must be specific to the actual candidate profile (no hallucination)
- Must not be templated (all identical strings = penalized)
- Must honestly reflect rank position
- Should mention concrete facts: years of experience, specific skills, location, signals
"""

from datetime import date
from dateutil.parser import parse as parse_date

TODAY = date(2026, 5, 28)


def days_since(date_str):
    if not date_str:
        return 9999
    try:
        return (TODAY - parse_date(date_str).date()).days
    except Exception:
        return 9999


def generate_reasoning(candidate, features, score, rank):
    """
    Generate a 1-2 sentence specific reasoning string for a candidate.
    
    This is NOT templated — it reads actual profile data to generate
    factual, candidate-specific text.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    name = profile.get("anonymized_name", "Candidate")
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Pick top relevant skills (advanced/expert with high endorsements)
    top_skills = sorted(
        [s for s in skills_list if s.get("proficiency") in ("advanced", "expert")],
        key=lambda s: s.get("endorsements", 0),
        reverse=True
    )[:3]
    top_skill_names = [s["name"] for s in top_skills]

    # Recency
    days_inactive = days_since(signals.get("last_active_date"))
    if days_inactive <= 7:
        activity_str = "active within the last week"
    elif days_inactive <= 30:
        activity_str = f"active {days_inactive} days ago"
    elif days_inactive <= 90:
        activity_str = f"last active ~{days_inactive // 30} months ago"
    else:
        activity_str = f"last active {days_inactive // 30} months ago (stale)"

    # Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:
        notice_str = f"immediately available ({notice}-day notice)"
    elif notice <= 30:
        notice_str = f"{notice}-day notice (within buyout range)"
    elif notice <= 60:
        notice_str = f"{notice}-day notice period"
    else:
        notice_str = f"long notice period ({notice} days — risk)"

    # Production signals from career
    ai_keywords = ["vector", "embedding", "retrieval", "ranking", "nlp", "llm", "rag",
                   "recommendation", "search", "fine-tun", "transformer"]
    production_evidence = []
    for job in career[:3]:
        desc = job.get("description", "").lower()
        hits = [kw for kw in ai_keywords if kw in desc]
        if hits:
            production_evidence.append(f"{job.get('title')} at {job.get('company')} ({', '.join(hits[:2])})")

    # Build location string
    location_str = f"{location}, {country}" if location and country else location or country or "location unknown"

    # Build the reasoning sentence
    parts = []

    # Sentence 1: Core qualifications
    skill_str = f"{', '.join(top_skill_names)}" if top_skill_names else "ML/AI skills"
    prod_str = f"; shipped {production_evidence[0]}" if production_evidence else ""
    parts.append(
        f"{yoe:.1f} yrs experience as {title} ({company}); strong on {skill_str}{prod_str}."
    )

    # Sentence 2: Signals and concerns
    concerns = []
    if days_inactive > 90:
        concerns.append(f"profile inactive {days_inactive // 30}mo")
    if notice > 60:
        concerns.append(f"{notice}d notice")
    if signals.get("recruiter_response_rate", 1) < 0.2:
        concerns.append("low recruiter response rate")
    if features.get("is_pure_consulting"):
        concerns.append("entire career at consulting firms")
    if features.get("location_fit", 1) < 0.5:
        concerns.append(f"based in {location_str}, relocation unclear")

    positives = []
    if signals.get("open_to_work_flag"):
        positives.append("open to work")
    if signals.get("github_activity_score", 0) > 50:
        positives.append(f"github score {signals['github_activity_score']:.0f}/100")
    if signals.get("saved_by_recruiters_30d", 0) > 3:
        positives.append(f"saved by {signals['saved_by_recruiters_30d']} recruiters recently")
    if features.get("location_fit", 0) >= 0.9:
        positives.append(f"based in {location_str}")

    signal_parts = []
    if positives:
        signal_parts.append(", ".join(positives[:2]))
    if concerns:
        signal_parts.append(f"concerns: {', '.join(concerns)}")
    signal_parts.append(notice_str)
    signal_parts.append(activity_str)

    parts.append(" | ".join(signal_parts) + ".")

    return " ".join(parts)
