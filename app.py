"""
app.py - Streamlit sandbox for Redrob Candidate Ranker
Judges can paste candidate JSON and see ranking output live.
"""

import streamlit as st
import json
import sys
import os

# Add current dir to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from features import extract_features
from score import compute_final_score
from reasoning import generate_reasoning

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.markdown("**India RUNS Hackathon — Data & AI Challenge**")
st.markdown("---")

st.markdown("""
### How it works
This system ranks candidates for a **Senior AI Engineer** role using:
- **TF-IDF Semantic Similarity** — matches candidate career text against the JD
- **Skill Match Scoring** — checks for 30+ required AI/ML skills with proficiency weighting  
- **Production AI Evidence** — detects real deployed systems in career history
- **Behavioral Signals** — recency, availability, notice period, response rate
- **Honeypot Detection** — filters out impossible/fake profiles
""")

st.markdown("---")

# Sample candidate for demo
SAMPLE_CANDIDATE = {
    "candidate_id": "DEMO_001",
    "profile": {
        "anonymized_name": "Demo Candidate",
        "headline": "Senior ML Engineer | RAG Systems | Vector Search | LLMs",
        "summary": "6 years building production ML systems. Shipped embedding-based retrieval at scale using FAISS and Pinecone. Fine-tuned LLMs with LoRA. Built ranking systems evaluated with NDCG.",
        "location": "Hyderabad",
        "country": "India",
        "years_of_experience": 6.5,
        "current_title": "Senior ML Engineer",
        "current_company": "AI Startup",
        "current_company_size": "51-200",
        "current_industry": "Artificial Intelligence"
    },
    "career_history": [
        {
            "company": "AI Startup",
            "title": "Senior ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 29,
            "is_current": True,
            "industry": "AI",
            "company_size": "51-200",
            "description": "Built production RAG system using FAISS vector database and sentence-transformers. Deployed ranking pipeline serving 10M queries/day. Fine-tuned BERT with LoRA for document retrieval. Evaluated with NDCG@10 and MRR metrics."
        },
        {
            "company": "Product Company",
            "title": "ML Engineer",
            "start_date": "2019-06-01",
            "end_date": "2021-12-31",
            "duration_months": 30,
            "is_current": False,
            "industry": "SaaS",
            "company_size": "201-500",
            "description": "Built recommendation system using collaborative filtering and embedding models. Implemented BM25 hybrid search. Shipped A/B testing framework for model evaluation."
        }
    ],
    "skills": [
        {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 45},
        {"name": "sentence-transformers", "proficiency": "advanced", "duration_months": 24, "endorsements": 12},
        {"name": "FAISS", "proficiency": "advanced", "duration_months": 20, "endorsements": 8},
        {"name": "RAG", "proficiency": "advanced", "duration_months": 18, "endorsements": 10},
        {"name": "LLM fine-tuning", "proficiency": "intermediate", "duration_months": 12, "endorsements": 5},
        {"name": "NDCG", "proficiency": "advanced", "duration_months": 24, "endorsements": 7},
        {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36, "endorsements": 15}
    ],
    "education": [
        {"degree": "B.Tech", "field": "Computer Science", "institution": "IIT Hyderabad", 
         "tier": "tier_1", "start_year": 2015, "end_year": 2019}
    ],
    "certifications": [
        {"name": "AWS Machine Learning Specialty", "issuer": "Amazon", "year": 2021}
    ],
    "redrob_signals": {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 4,
        "notice_period_days": 30,
        "interview_completion_rate": 0.90,
        "offer_acceptance_rate": 0.75,
        "profile_completeness_score": 92,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "github_activity_score": 72,
        "saved_by_recruiters_30d": 8,
        "applications_submitted_30d": 3,
        "willing_to_relocate": True,
        "skill_assessment_scores": {
            "Python": 91,
            "sentence-transformers": 85,
            "FAISS": 80
        }
    }
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input — Candidate JSON")
    st.markdown("Paste a candidate profile or use the sample below:")
    
    candidate_json = st.text_area(
        "Candidate JSON",
        value=json.dumps(SAMPLE_CANDIDATE, indent=2),
        height=500,
        label_visibility="collapsed"
    )
    
    # Similarity score slider (simulates TF-IDF output)
    st.markdown("**Semantic Similarity Score** (from TF-IDF pipeline):")
    sim_score = st.slider(
        "similarity", 0.0, 1.0, 0.72,
        help="In production this is computed automatically by embed.py",
        label_visibility="collapsed"
    )
    
    run_button = st.button("🚀 Rank this Candidate", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Output — Ranking Result")
    
    if run_button:
        try:
            candidate = json.loads(candidate_json)
            features = extract_features(candidate)
            score = compute_final_score(features, sim_score)
            reasoning = generate_reasoning(candidate, features, score, 1)

            # Score display
            score_color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
            st.markdown(f"### Final Score: :{score_color}[{score:.4f}]")
            
            # Status
            if features["is_honeypot"]:
                st.error("🚨 HONEYPOT DETECTED — Excluded from ranking")
            elif features["is_disqualified_title"]:
                st.warning("⚠️ Disqualified title — Low score applied")
            elif features["is_pure_consulting"]:
                st.warning("⚠️ Pure consulting background — Penalized")
            elif score > 0.75:
                st.success("✅ Strong candidate — Would appear in top 20")
            elif score > 0.60:
                st.info("📌 Moderate fit — Would appear in top 50")
            else:
                st.warning("⚠️ Weak fit — Would not appear in top 100")

            st.markdown("---")
            st.markdown("**🧠 Reasoning:**")
            st.info(reasoning)

            st.markdown("---")
            st.markdown("**📈 Score Breakdown:**")

            from score import compute_behavioral_score
            behavioral = compute_behavioral_score(features)

            breakdown = {
                "Semantic Similarity (30%)": f"{sim_score:.3f}",
                "Required Skill Match (20%)": f"{features.get('required_skill_score', 0):.3f}",
                "Production AI Evidence (15%)": f"{features.get('production_ai_score', 0):.3f}",
                "Behavioral Score (15%)": f"{behavioral:.3f}",
                "Experience Fit (8%)": f"{features.get('exp_fit_score', 0):.3f}",
                "Product Company Ratio (5%)": f"{min(features.get('product_company_ratio', 0), 1.0):.3f}",
                "Location Fit (4%)": f"{features.get('location_fit', 0):.3f}",
            }

            for key, val in breakdown.items():
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"• {key}")
                col_b.markdown(f"**{val}**")

            st.markdown("---")
            st.markdown("**🔍 Feature Flags:**")
            flags = {
                "Is Honeypot": features.get("is_honeypot", False),
                "Disqualified Title": features.get("is_disqualified_title", False),
                "Pure Consulting": features.get("is_pure_consulting", False),
                "Open to Work": features.get("open_to_work", 0) == 1.0,
                "Title is Technical": features.get("title_is_technical", False),
            }
            for flag, val in flags.items():
                icon = "✅" if val else "❌"
                st.markdown(f"{icon} {flag}")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.markdown("👈 Paste a candidate JSON and click **Rank this Candidate**")
        st.markdown("""
        **What you'll see:**
        - Final composite score (0-1)
        - Score breakdown by component  
        - Candidate-specific reasoning
        - Feature flags (honeypot, disqualified, etc.)
        """)

st.markdown("---")
st.markdown("""
**Architecture:** TF-IDF Vectorizer (8000 features, bigrams) → Cosine Similarity vs JD → 
Weighted combination with 20+ structured features → Behavioral signal multiplier → Final ranked CSV

**GitHub:** https://github.com/workingsharansh08/redrob-candidate-ranker
""")
