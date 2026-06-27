"""
features.py
-----------
Extracts a structured feature dictionary from a raw candidate JSON record.
Run once per candidate during precompute. Fast, no ML involved.
"""

from datetime import date
from dateutil.parser import parse as parse_date

# ── JD-derived constants ───────────────────────────────────────────────────────

# Skills the JD says are REQUIRED (must-haves)
REQUIRED_SKILLS = {
    "sentence-transformers", "embeddings", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "hybrid search", "retrieval", "ranking", "nlp", "python",
    "ndcg", "mrr", "map", "a/b testing", "fine-tuning", "lora", "qlora",
    "rag", "transformer", "bert", "recommendation system", "search",
    "learning to rank", "xgboost", "bm25", "dense retrieval",
    "peft", "llm", "large language model", "reranking"
}

# Skills that are NICE TO HAVE (bonus)
NICE_SKILLS = {
    "distributed systems", "inference optimization", "open-source",
    "hr tech", "recruiting", "marketplace", "spark", "kafka",
    "pytorch", "tensorflow", "huggingface", "mlops", "kubernetes", "docker"
}

# Hard disqualifiers from the JD's "we explicitly do NOT want" section
DISQUALIFIED_TITLES = {
    "marketing manager", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "graphic designer", "operations manager",
    "customer support", "business analyst", "project manager",
    "frontend engineer", "ui designer", "ux designer", "sales",
    "content writer", "seo", "recruiter"
}

# Consulting firms — JD explicitly says "if your ENTIRE career is here, not a fit"
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree"
}

# Good company types (product companies > consulting)
PRODUCT_COMPANY_SIGNALS = {
    "startup", "saas", "product", "ai", "ml", "fintech", "edtech",
    "healthtech", "ecommerce", "platform", "b2b", "series"
}

TODAY = date(2026, 5, 28)  # Use dataset reference date (last active dates go up to May 2026)


def days_since(date_str):
    """How many days ago was this date? Returns 9999 if missing."""
    if not date_str:
        return 9999
    try:
        return (TODAY - parse_date(date_str).date()).days
    except Exception:
        return 9999


def is_honeypot(candidate):
    """
    Detect candidates with impossible/contradictory profiles.
    Returns True if honeypot (should be excluded).
    
    Honeypot patterns:
    1. Claims experience at company that didn't exist yet
    2. Expert in 10+ skills with 0 months used
    3. Experience years >> sum of career history months
    4. Education end_year in the future but already has 10+ years experience
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    stated_exp_years = profile.get("years_of_experience", 0)

    # Check 1: Expert skills with 0 months duration (impossible)
    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_experts) >= 3:
        return True

    # Check 2: Career history total months vs stated years (big discrepancy)
    total_career_months = sum(c.get("duration_months", 0) for c in career)
    if total_career_months > 0:
        career_years = total_career_months / 12.0
        # If stated experience is >4 years more than career history, suspicious
        if stated_exp_years > career_years + 4:
            return True

    # Check 3: Education end_year impossible
    for edu in education:
        end_year = edu.get("end_year", 0)
        start_year = edu.get("start_year", 0)
        if end_year > 2026:
            return True
        if start_year > 0 and end_year > 0 and (end_year - start_year) > 10:
            return True

    # Check 4: Company tenure impossibility — worked there longer than company exists
    for job in career:
        dur = job.get("duration_months", 0)
        company = job.get("company", "").lower()
        start_str = job.get("start_date", "")
        if dur > 0 and start_str:
            try:
                start = parse_date(start_str).date()
                if start.year < 1990 and dur > 12:
                    return True  # Suspiciously old start with long tenure
            except Exception:
                pass

    return False


def extract_features(candidate):
    """
    Main feature extraction function.
    Returns a dict with all numeric features for scoring.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})
    certs = candidate.get("certifications", [])

    features = {}
    features["candidate_id"] = candidate["candidate_id"]

    # ── 1. HARD FILTER FLAGS ───────────────────────────────────────────────────

    # Is the current title clearly non-technical / disqualified?
    current_title = profile.get("current_title", "").lower()
    features["is_disqualified_title"] = any(
        d in current_title for d in DISQUALIFIED_TITLES
    )

    # Is entire career at consulting firms only?
    all_companies = [c.get("company", "").lower() for c in career]
    consulting_jobs = sum(
        1 for co in all_companies
        if any(cf in co for cf in CONSULTING_FIRMS)
    )
    features["is_pure_consulting"] = (
        len(all_companies) > 0 and consulting_jobs == len(all_companies)
    )

    # Honeypot flag
    features["is_honeypot"] = is_honeypot(candidate)

    # Location fit (JD: Pune/Noida/Hyderabad/Mumbai/Delhi NCR or willing to relocate)
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    preferred_locations = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "bangalore", "bengaluru", "gurgaon"]
    in_preferred_location = any(loc in location for loc in preferred_locations)
    in_india = country == "india"
    willing_to_relocate = signals.get("willing_to_relocate", False)
    features["location_fit"] = 1.0 if in_preferred_location else (0.7 if (in_india and willing_to_relocate) else (0.4 if willing_to_relocate else 0.0))

    # ── 2. EXPERIENCE FEATURES ────────────────────────────────────────────────

    yoe = profile.get("years_of_experience", 0)
    features["years_of_experience"] = yoe

    # JD says 5-9 years ideal; 4 is ok; 10+ is ok if signals strong
    if 5 <= yoe <= 9:
        features["exp_fit_score"] = 1.0
    elif 4 <= yoe < 5:
        features["exp_fit_score"] = 0.85
    elif 9 < yoe <= 12:
        features["exp_fit_score"] = 0.80
    elif 3 <= yoe < 4:
        features["exp_fit_score"] = 0.60
    elif yoe > 12:
        features["exp_fit_score"] = 0.65
    else:
        features["exp_fit_score"] = 0.20

    # Product company experience (vs pure consulting)
    product_months = 0
    for job in career:
        industry = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        desc = job.get("description", "").lower()
        is_product = any(p in industry or p in company or p in desc for p in PRODUCT_COMPANY_SIGNALS)
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        if is_product and not is_consulting:
            product_months += job.get("duration_months", 0)
    features["product_company_months"] = product_months
    features["product_company_ratio"] = product_months / max(yoe * 12, 1)

    # ── 3. SKILL FEATURES ─────────────────────────────────────────────────────

    skill_names_lower = {s["name"].lower() for s in skills}
    skill_duration = {s["name"].lower(): s.get("duration_months", 0) for s in skills}
    skill_proficiency = {s["name"].lower(): s.get("proficiency", "") for s in skills}
    skill_assessment = {k.lower(): v for k, v in signals.get("skill_assessment_scores", {}).items()}

    # Required skill hits (weighted by proficiency)
    required_hit_score = 0.0
    proficiency_weights = {"beginner": 0.3, "intermediate": 0.7, "advanced": 1.0, "expert": 1.2}
    for req in REQUIRED_SKILLS:
        for skill_name in skill_names_lower:
            if req in skill_name or skill_name in req:
                prof = skill_proficiency.get(skill_name, "beginner")
                weight = proficiency_weights.get(prof, 0.3)
                # Bonus for actual assessment score
                assess_score = skill_assessment.get(skill_name, -1)
                if assess_score > 0:
                    weight *= (0.5 + assess_score / 100.0)
                required_hit_score += weight
                break

    features["required_skill_score"] = min(required_hit_score / len(REQUIRED_SKILLS), 1.0)

    # Nice-to-have skills
    nice_hits = sum(1 for n in NICE_SKILLS if any(n in s for s in skill_names_lower))
    features["nice_skill_score"] = nice_hits / len(NICE_SKILLS)

    # Skill depth — total months in ML/AI relevant skills
    ai_skill_months = sum(
        dur for name, dur in skill_duration.items()
        if any(req in name or name in req for req in REQUIRED_SKILLS)
    )
    features["ai_skill_months"] = ai_skill_months

    # ── 4. CAREER QUALITY ─────────────────────────────────────────────────────

    # Has shipped production AI/ML systems? Look in job descriptions
    ai_production_signals = [
        "production", "deployed", "served", "serving", "scaled", "shipped",
        "vector", "embedding", "retrieval", "ranking", "recommendation",
        "search", "nlp", "llm", "fine-tun", "rag", "pipeline at scale"
    ]
    production_score = 0
    for job in career:
        desc = (job.get("description", "") + " " + job.get("title", "")).lower()
        hits = sum(1 for s in ai_production_signals if s in desc)
        production_score += hits
    features["production_ai_score"] = min(production_score / 10.0, 1.0)

    # Job title quality (current role should be technical AI/ML)
    ml_titles = ["ml", "machine learning", "ai", "artificial intelligence", "data scientist",
                 "nlp", "research scientist", "applied scientist", "ai engineer", "ml engineer",
                 "data engineer", "software engineer", "backend engineer", "platform engineer"]
    features["title_is_technical"] = any(t in current_title for t in ml_titles)

    # Education tier
    edu_tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25}
    best_edu = max(
        (edu_tier_scores.get(e.get("tier", "tier_4"), 0.25) for e in education),
        default=0.25
    )
    features["education_tier_score"] = best_edu

    # Certifications in ML/AI
    ml_cert_keywords = ["aws", "gcp", "azure", "tensorflow", "pytorch", "deep learning",
                        "machine learning", "data science", "nlp", "hugging"]
    ml_certs = sum(1 for c in certs if any(k in c.get("name", "").lower() for k in ml_cert_keywords))
    features["ml_cert_count"] = ml_certs

    # ── 5. BEHAVIORAL SIGNALS (the 23) ────────────────────────────────────────

    # Recency / availability
    days_inactive = days_since(signals.get("last_active_date"))
    if days_inactive <= 7:
        features["recency_score"] = 1.0
    elif days_inactive <= 30:
        features["recency_score"] = 0.9
    elif days_inactive <= 90:
        features["recency_score"] = 0.7
    elif days_inactive <= 180:
        features["recency_score"] = 0.4
    else:
        features["recency_score"] = 0.1

    features["open_to_work"] = 1.0 if signals.get("open_to_work_flag") else 0.5

    # Responsiveness — JD cares: "active on Redrob platform, actually talk to them"
    response_rate = signals.get("recruiter_response_rate", 0)
    avg_response_hours = signals.get("avg_response_time_hours", 999)
    features["responsiveness_score"] = (
        response_rate * 0.7 +
        max(0, 1.0 - avg_response_hours / 168.0) * 0.3  # 168h = 1 week
    )

    # Notice period — JD wants sub-30 days ideally, can buy out 30 days
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:
        features["notice_score"] = 1.0
    elif notice <= 30:
        features["notice_score"] = 0.90
    elif notice <= 60:
        features["notice_score"] = 0.65
    elif notice <= 90:
        features["notice_score"] = 0.40
    else:
        features["notice_score"] = 0.15

    # Interview reliability
    icr = signals.get("interview_completion_rate", 0.5)
    features["interview_reliability"] = icr

    # Offer acceptance (signals intent)
    oar = signals.get("offer_acceptance_rate", -1)
    features["offer_acceptance"] = oar if oar >= 0 else 0.5

    # Profile quality
    features["profile_completeness"] = signals.get("profile_completeness_score", 50) / 100.0
    features["verified"] = (
        0.5 * int(signals.get("verified_email", False)) +
        0.3 * int(signals.get("verified_phone", False)) +
        0.2 * int(signals.get("linkedin_connected", False))
    )

    # GitHub activity (JD values open-source contributions)
    gh_score = signals.get("github_activity_score", -1)
    features["github_score"] = gh_score / 100.0 if gh_score >= 0 else 0.1

    # Recruiter interest (market validation of candidate)
    features["market_validation"] = min(signals.get("saved_by_recruiters_30d", 0) / 10.0, 1.0)

    # Profile activity
    features["applications_active"] = min(signals.get("applications_submitted_30d", 0) / 5.0, 1.0)

    return features
