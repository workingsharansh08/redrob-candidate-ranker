"""
score.py
--------
Combines semantic embedding similarity + structured features into a final score.
This is the core scoring logic — the "brain" of the ranker.

Weights are tuned based on what the JD emphasizes most.
"""

import numpy as np


# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
# These reflect priority order from the JD and submission spec:
# NDCG@10 is 50% of score → top 10 must be perfect → semantic + skill match drive this

WEIGHTS = {
    "semantic_similarity":   0.30,  # How well candidate text matches JD text (embedding cosine sim)
    "required_skill_score":  0.20,  # Hits on JD's "things you absolutely need"
    "production_ai_score":   0.15,  # Evidence of shipping AI systems in production
    "behavioral_score":      0.15,  # Availability, responsiveness, recency
    "exp_fit_score":         0.08,  # Years experience fit (5-9 ideal)
    "product_company_ratio": 0.05,  # Product company vs consulting ratio
    "location_fit":          0.04,  # Location / relocation fit
    "education_tier_score":  0.02,  # Education tier (minor signal)
    "github_score":          0.01,  # Open source activity (JD mentions as nice-to-have)
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


def compute_behavioral_score(features):
    """
    Composite behavioral availability score.
    A great candidate who is unreachable is worthless for hiring.
    """
    return (
        features.get("recency_score", 0.5) * 0.35 +
        features.get("open_to_work", 0.5) * 0.20 +
        features.get("responsiveness_score", 0.5) * 0.20 +
        features.get("notice_score", 0.5) * 0.15 +
        features.get("interview_reliability", 0.5) * 0.10
    )


def compute_final_score(features, semantic_similarity):
    """
    Main scoring function.
    
    Args:
        features: dict from features.extract_features()
        semantic_similarity: float [0,1] cosine similarity vs JD embedding
    
    Returns:
        float: final score [0,1]
    """

    # Hard disqualifications — these return near-zero immediately
    if features.get("is_honeypot"):
        return 0.001  # Never in top 100

    if features.get("is_disqualified_title") and features.get("required_skill_score", 0) < 0.15:
        # Marketing Managers, Accountants, Civil Engineers etc. with no real ML skills
        return 0.002

    if features.get("is_pure_consulting") and features.get("required_skill_score", 0) < 0.20:
        # Entire career at TCS/Wipro/Infosys with no real ML skills
        return 0.003

    # Compute behavioral composite
    behavioral_score = compute_behavioral_score(features)

    # Build weighted sum
    raw_score = (
        WEIGHTS["semantic_similarity"]   * float(semantic_similarity) +
        WEIGHTS["required_skill_score"]  * features.get("required_skill_score", 0) +
        WEIGHTS["production_ai_score"]   * features.get("production_ai_score", 0) +
        WEIGHTS["behavioral_score"]      * behavioral_score +
        WEIGHTS["exp_fit_score"]         * features.get("exp_fit_score", 0) +
        WEIGHTS["product_company_ratio"] * min(features.get("product_company_ratio", 0), 1.0) +
        WEIGHTS["location_fit"]          * features.get("location_fit", 0) +
        WEIGHTS["education_tier_score"]  * features.get("education_tier_score", 0) +
        WEIGHTS["github_score"]          * features.get("github_score", 0)
    )

    # Bonus multipliers (these can push borderline candidates up)
    multiplier = 1.0

    # Active job seeker with strong engagement
    if features.get("open_to_work") == 1.0 and features.get("recency_score", 0) >= 0.9:
        multiplier *= 1.05

    # High market validation — recruiters are already saving this candidate
    if features.get("market_validation", 0) > 0.5:
        multiplier *= 1.03

    # Nice skills bonus
    if features.get("nice_skill_score", 0) > 0.3:
        multiplier *= 1.02

    final = min(raw_score * multiplier, 1.0)
    return round(final, 6)


def rank_candidates(all_features, semantic_similarities):
    """
    Given feature dicts and similarity scores, compute final scores and rank.
    
    Returns list of (candidate_id, final_score, features) sorted best-first.
    """
    scored = []
    for features, sim in zip(all_features, semantic_similarities):
        score = compute_final_score(features, sim)
        scored.append((features["candidate_id"], score, features))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
