"""
score.py - ENHANCED VERSION
Updated weights to include company tier, recency, industry relevance,
career growth, title progression
"""

import numpy as np

WEIGHTS = {
    "semantic_similarity":        0.28,
    "required_skill_score":       0.18,
    "recency_weighted_ai_score":  0.12,
    "production_ai_score":        0.10,
    "behavioral_score":           0.12,
    "exp_fit_score":              0.07,
    "company_tier_score":         0.05,
    "product_company_ratio":      0.03,
    "industry_relevance":         0.02,
    "title_progression_score":    0.02,
    "location_fit":               0.01,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


def compute_behavioral_score(features):
    return (
        features.get("recency_score", 0.5) * 0.35 +
        features.get("open_to_work", 0.5) * 0.20 +
        features.get("responsiveness_score", 0.5) * 0.20 +
        features.get("notice_score", 0.5) * 0.15 +
        features.get("interview_reliability", 0.5) * 0.10
    )


def compute_final_score(features, semantic_similarity):
    # Hard disqualifications
    if features.get("is_honeypot"):
        return 0.001

    if features.get("is_disqualified_title") and features.get("required_skill_score", 0) < 0.15:
        return 0.002

    if features.get("is_pure_consulting") and features.get("required_skill_score", 0) < 0.20:
        return 0.003

    # Hard filter: less than 3 years experience
    if features.get("years_of_experience", 0) < 3:
        return 0.004

    behavioral_score = compute_behavioral_score(features)

    raw_score = (
        WEIGHTS["semantic_similarity"]        * float(semantic_similarity) +
        WEIGHTS["required_skill_score"]       * features.get("required_skill_score", 0) +
        WEIGHTS["recency_weighted_ai_score"]  * features.get("recency_weighted_ai_score", 0) +
        WEIGHTS["production_ai_score"]        * features.get("production_ai_score", 0) +
        WEIGHTS["behavioral_score"]           * behavioral_score +
        WEIGHTS["exp_fit_score"]              * features.get("exp_fit_score", 0) +
        WEIGHTS["company_tier_score"]         * features.get("company_tier_score", 0.3) +
        WEIGHTS["product_company_ratio"]      * min(features.get("product_company_ratio", 0), 1.0) +
        WEIGHTS["industry_relevance"]         * features.get("industry_relevance", 0.4) +
        WEIGHTS["title_progression_score"]    * features.get("title_progression_score", 0.5) +
        WEIGHTS["location_fit"]               * features.get("location_fit", 0)
    )

    # Bonus multipliers
    multiplier = 1.0

    # Tier 1 company bonus
    if features.get("company_tier_score", 0) >= 0.9:
        multiplier *= 1.06

    # Active job seeker
    if features.get("open_to_work") == 1.0 and features.get("recency_score", 0) >= 0.9:
        multiplier *= 1.04

    # Strong market validation
    if features.get("market_validation", 0) > 0.5:
        multiplier *= 1.03

    # Nice skills bonus
    if features.get("nice_skill_score", 0) > 0.3:
        multiplier *= 1.02

    # GitHub activity bonus
    if features.get("github_score", 0) > 0.6:
        multiplier *= 1.02

    # Career growth bonus
    if features.get("career_growth", 0) > 0.6:
        multiplier *= 1.02

    final = min(raw_score * multiplier, 1.0)
    return round(final, 6)


def rank_candidates(all_features, semantic_similarities):
    scored = []
    for features, sim in zip(all_features, semantic_similarities):
        score = compute_final_score(features, sim)
        scored.append((features["candidate_id"], score, features))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
