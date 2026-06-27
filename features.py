"""
features.py - ENHANCED VERSION
Adds: career recency weighting, company tier scoring, 
skill recency, job title progression, industry relevance
"""

from datetime import date
from dateutil.parser import parse as parse_date

# ── JD-derived constants ───────────────────────────────────────────────────────

REQUIRED_SKILLS = {
    "sentence-transformers", "embeddings", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "hybrid search", "retrieval", "ranking", "nlp", "python",
    "ndcg", "mrr", "map", "a/b testing", "fine-tuning", "lora", "qlora",
    "rag", "transformer", "bert", "recommendation system", "search",
    "learning to rank", "xgboost", "bm25", "dense retrieval",
    "peft", "llm", "large language model", "reranking", "vector search",
    "semantic search", "neural ranking", "information retrieval",
    "pytorch", "tensorflow", "huggingface", "machine learning"
}

NICE_SKILLS = {
    "distributed systems", "inference optimization", "open-source",
    "hr tech", "recruiting", "marketplace", "spark", "kafka",
    "mlops", "kubernetes", "docker", "data engineering", "airflow"
}

DISQUALIFIED_TITLES = {
    "marketing manager", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "graphic designer", "operations manager",
    "customer support", "business analyst", "project manager",
    "ui designer", "ux designer", "sales", "content writer",
    "seo", "recruiter", "financial analyst", "product manager"
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "persistent", "mastech", "niit technologies"
}

# Tier 1 — Top Indian product/AI companies (big boost)
TIER1_COMPANIES = {
    "swiggy", "zomato", "flipkart", "meesho", "razorpay", "cred",
    "zepto", "groww", "phonepe", "juspay", "paytm", "ola", "uber",
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
    "openai", "anthropic", "deepmind", "nvidia", "salesforce",
    "atlassian", "freshworks", "zoho", "cleartax", "sharechat",
    "dream11", "mpl", "unacademy", "byju", "vedantu", "upgrad",
    "lenskart", "nykaa", "myntra", "bigbasket", "dunzo", "rapido",
    "slice", "fi money", "jupiter", "smallcase", "zerodha", "kite"
}

# Tier 2 — Good product companies (moderate boost)
TIER2_COMPANIES = {
    "startup", "saas", "series a", "series b", "series c",
    "ai company", "ml company", "fintech", "edtech", "healthtech",
    "proptech", "insurtech", "legaltech", "hrtech", "martech"
}

PRODUCT_COMPANY_SIGNALS = {
    "startup", "saas", "product", "ai", "ml", "fintech", "edtech",
    "healthtech", "ecommerce", "platform", "b2b", "series"
}

# High relevance industries
RELEVANT_INDUSTRIES = {
    "artificial intelligence", "machine learning", "data science",
    "technology", "software", "internet", "e-commerce", "fintech",
    "edtech", "healthtech", "saas", "cloud", "analytics",
    "information technology", "computer software"
}

# Title progression scoring
TITLE_SENIORITY = {
    "intern": 0.1, "junior": 0.3, "associate": 0.4,
    "engineer": 0.5, "developer": 0.5, "scientist": 0.6,
    "senior": 0.8, "lead": 0.85, "staff": 0.9,
    "principal": 0.95, "director": 0.85, "head": 0.85,
    "vp": 0.8, "chief": 0.75, "founder": 0.7
}

TODAY = date(2026, 5, 28)


def days_since(date_str):
    if not date_str:
        return 9999
    try:
        return (TODAY - parse_date(date_str).date()).days
    except Exception:
        return 9999


def is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    stated_exp_years = profile.get("years_of_experience", 0)

    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_experts) >= 3:
        return True

    total_career_months = sum(c.get("duration_months", 0) for c in career)
    if total_career_months > 0:
        career_years = total_career_months / 12.0
        if stated_exp_years > career_years + 4:
            return True

    for edu in education:
        end_year = edu.get("end_year", 0)
        start_year = edu.get("start_year", 0)
        if end_year > 2026:
            return True
        if start_year > 0 and end_year > 0 and (end_year - start_year) > 10:
            return True

    return False


def get_title_seniority_score(title):
    """Score a job title by seniority level"""
    title_lower = title.lower()
    best_score = 0.3
    for keyword, score in TITLE_SENIORITY.items():
        if keyword in title_lower:
            best_score = max(best_score, score)
    return best_score


def get_company_tier(company_name, industry=""):
    """Score company by tier"""
    company_lower = company_name.lower()
    industry_lower = industry.lower()

    # Check tier 1
    if any(t1 in company_lower for t1 in TIER1_COMPANIES):
        return 1.0

    # Check tier 2 signals
    if any(t2 in company_lower or t2 in industry_lower for t2 in TIER2_COMPANIES):
        return 0.75

    # Product company signals
    if any(p in company_lower or p in industry_lower for p in PRODUCT_COMPANY_SIGNALS):
        return 0.65

    # Consulting firms
    if any(cf in company_lower for cf in CONSULTING_FIRMS):
        return 0.2

    return 0.45  # Unknown company — neutral


def extract_features(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})
    certs = candidate.get("certifications", [])

    features = {}
    features["candidate_id"] = candidate["candidate_id"]

    # ── 1. HARD FILTER FLAGS ───────────────────────────────────────────────────
    current_title = profile.get("current_title", "").lower()
    features["is_disqualified_title"] = any(
        d in current_title for d in DISQUALIFIED_TITLES
    )

    all_companies = [c.get("company", "").lower() for c in career]
    consulting_jobs = sum(
        1 for co in all_companies
        if any(cf in co for cf in CONSULTING_FIRMS)
    )
    features["is_pure_consulting"] = (
        len(all_companies) > 0 and consulting_jobs == len(all_companies)
    )

    features["is_honeypot"] = is_honeypot(candidate)

    # Location fit
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    preferred_locations = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
                           "bangalore", "bengaluru", "gurgaon", "chennai", "kolkata"]
    in_preferred_location = any(loc in location for loc in preferred_locations)
    in_india = country == "india"
    willing_to_relocate = signals.get("willing_to_relocate", False)
    features["location_fit"] = (
        1.0 if in_preferred_location else
        0.7 if (in_india and willing_to_relocate) else
        0.4 if willing_to_relocate else 0.0
    )

    # ── 2. EXPERIENCE FEATURES ────────────────────────────────────────────────
    yoe = profile.get("years_of_experience", 0)
    features["years_of_experience"] = yoe

    # Hard filter: less than 3 years = very unlikely fit
    if yoe < 3:
        features["exp_fit_score"] = 0.05
    elif 5 <= yoe <= 9:
        features["exp_fit_score"] = 1.0
    elif 4 <= yoe < 5:
        features["exp_fit_score"] = 0.85
    elif 9 < yoe <= 12:
        features["exp_fit_score"] = 0.80
    elif 3 <= yoe < 4:
        features["exp_fit_score"] = 0.50
    elif yoe > 12:
        features["exp_fit_score"] = 0.65
    else:
        features["exp_fit_score"] = 0.20

    # ── 3. CAREER RECENCY WEIGHTING (NEW) ────────────────────────────────────
    # Recent ML/AI experience is worth much more than old experience
    ai_keywords = ["vector", "embedding", "retrieval", "ranking", "nlp", "llm",
                   "rag", "recommendation", "search", "fine-tun", "transformer",
                   "machine learning", "deep learning", "neural", "bert", "gpt",
                   "deployed", "production", "shipped", "served", "scaled"]

    recency_weighted_score = 0.0
    production_score = 0
    company_tier_scores = []
    title_progression = []

    for i, job in enumerate(career[:6]):
        desc = (job.get("description", "") + " " + job.get("title", "")).lower()
        company = job.get("company", "")
        industry = job.get("industry", "")
        job_title = job.get("title", "")
        duration = job.get("duration_months", 0)
        is_current = job.get("is_current", False)

        # Recency weight — current job = 1.0, previous = 0.7, older = 0.5, etc.
        recency_weight = 1.0 if is_current else max(0.3, 1.0 - i * 0.15)

        # AI/ML signal hits in this job
        ai_hits = sum(1 for kw in ai_keywords if kw in desc)
        recency_weighted_score += ai_hits * recency_weight

        # Production signals
        production_keywords = ["production", "deployed", "serving", "scaled",
                               "shipped", "real users", "million", "billion"]
        prod_hits = sum(1 for kw in production_keywords if kw in desc)
        production_score += prod_hits * recency_weight

        # Company tier
        tier = get_company_tier(company, industry)
        company_tier_scores.append(tier * recency_weight)

        # Title progression
        seniority = get_title_seniority_score(job_title)
        title_progression.append(seniority)

    features["recency_weighted_ai_score"] = min(recency_weighted_score / 20.0, 1.0)
    features["production_ai_score"] = min(production_score / 8.0, 1.0)
    features["company_tier_score"] = (
        sum(company_tier_scores) / len(company_tier_scores)
        if company_tier_scores else 0.3
    )

    # Title progression — is career going up?
    if len(title_progression) >= 2:
        # Current title seniority vs average of previous
        current_seniority = title_progression[0]
        prev_avg = sum(title_progression[1:]) / len(title_progression[1:])
        features["title_progression_score"] = min(current_seniority, 1.0)
        features["career_growth"] = max(0, current_seniority - prev_avg + 0.5)
    else:
        features["title_progression_score"] = get_title_seniority_score(
            profile.get("current_title", ""))
        features["career_growth"] = 0.5

    # Product company ratio
    product_months = 0
    total_months = sum(c.get("duration_months", 0) for c in career)
    for job in career:
        industry = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        desc = job.get("description", "").lower()
        is_product = any(
            p in industry or p in company or p in desc
            for p in PRODUCT_COMPANY_SIGNALS
        )
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        if is_product and not is_consulting:
            product_months += job.get("duration_months", 0)

    features["product_company_months"] = product_months
    features["product_company_ratio"] = product_months / max(total_months, 1)

    # ── 4. INDUSTRY RELEVANCE (NEW) ───────────────────────────────────────────
    current_industry = profile.get("current_industry", "").lower()
    industry_relevance = any(ri in current_industry for ri in RELEVANT_INDUSTRIES)
    features["industry_relevance"] = 1.0 if industry_relevance else 0.4

    # ── 5. SKILL FEATURES ─────────────────────────────────────────────────────
    skill_names_lower = {s["name"].lower() for s in skills}
    skill_duration = {s["name"].lower(): s.get("duration_months", 0) for s in skills}
    skill_proficiency = {s["name"].lower(): s.get("proficiency", "") for s in skills}
    skill_assessment = {
        k.lower(): v for k, v in signals.get("skill_assessment_scores", {}).items()
    }

    # Skill recency — approximate from last_used if available
    proficiency_weights = {
        "beginner": 0.3, "intermediate": 0.7, "advanced": 1.0, "expert": 1.2
    }

    required_hit_score = 0.0
    for req in REQUIRED_SKILLS:
        for skill_name in skill_names_lower:
            if req in skill_name or skill_name in req:
                prof = skill_proficiency.get(skill_name, "beginner")
                weight = proficiency_weights.get(prof, 0.3)
                assess_score = skill_assessment.get(skill_name, -1)
                if assess_score > 0:
                    weight *= (0.5 + assess_score / 100.0)
                # Bonus for longer duration (more experienced with skill)
                dur = skill_duration.get(skill_name, 0)
                if dur > 24:
                    weight *= 1.1
                required_hit_score += weight
                break

    features["required_skill_score"] = min(
        required_hit_score / len(REQUIRED_SKILLS), 1.0
    )

    nice_hits = sum(
        1 for n in NICE_SKILLS if any(n in s for s in skill_names_lower)
    )
    features["nice_skill_score"] = nice_hits / len(NICE_SKILLS)

    ai_skill_months = sum(
        dur for name, dur in skill_duration.items()
        if any(req in name or name in req for req in REQUIRED_SKILLS)
    )
    features["ai_skill_months"] = ai_skill_months

    # Title technical check
    ml_titles = ["ml", "machine learning", "ai", "artificial intelligence",
                 "data scientist", "nlp", "research scientist", "applied scientist",
                 "ai engineer", "ml engineer", "data engineer", "software engineer",
                 "backend engineer", "platform engineer", "search engineer",
                 "recommendation", "ranking engineer", "applied researcher"]
    features["title_is_technical"] = any(t in current_title for t in ml_titles)

    # Education tier
    edu_tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25}
    best_edu = max(
        (edu_tier_scores.get(e.get("tier", "tier_4"), 0.25) for e in education),
        default=0.25
    )
    features["education_tier_score"] = best_edu

    # ML certifications
    ml_cert_keywords = ["aws", "gcp", "azure", "tensorflow", "pytorch",
                        "deep learning", "machine learning", "data science",
                        "nlp", "hugging", "google cloud", "databricks"]
    ml_certs = sum(
        1 for c in certs
        if any(k in c.get("name", "").lower() for k in ml_cert_keywords)
    )
    features["ml_cert_count"] = ml_certs

    # ── 6. BEHAVIORAL SIGNALS ─────────────────────────────────────────────────
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

    response_rate = signals.get("recruiter_response_rate", 0)
    avg_response_hours = signals.get("avg_response_time_hours", 999)
    features["responsiveness_score"] = (
        response_rate * 0.7 +
        max(0, 1.0 - avg_response_hours / 168.0) * 0.3
    )

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

    features["interview_reliability"] = signals.get("interview_completion_rate", 0.5)
    oar = signals.get("offer_acceptance_rate", -1)
    features["offer_acceptance"] = oar if oar >= 0 else 0.5
    features["profile_completeness"] = (
        signals.get("profile_completeness_score", 50) / 100.0
    )
    features["verified"] = (
        0.5 * int(signals.get("verified_email", False)) +
        0.3 * int(signals.get("verified_phone", False)) +
        0.2 * int(signals.get("linkedin_connected", False))
    )

    gh_score = signals.get("github_activity_score", -1)
    features["github_score"] = gh_score / 100.0 if gh_score >= 0 else 0.1
    features["market_validation"] = min(
        signals.get("saved_by_recruiters_30d", 0) / 10.0, 1.0
    )
    features["applications_active"] = min(
        signals.get("applications_submitted_30d", 0) / 5.0, 1.0
    )

    return features
