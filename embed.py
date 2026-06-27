"""
embed.py - NO INTERNET REQUIRED VERSION
Uses TF-IDF similarity instead of sentence-transformers.
Fast, offline, no model download needed.
"""

import argparse
import json
import gzip
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JD_TEXT = """
Senior AI Engineer founding team Series A startup talent intelligence platform.
Production embeddings retrieval systems sentence-transformers vector database FAISS 
pinecone weaviate qdrant milvus opensearch elasticsearch hybrid search dense retrieval.
Strong Python NLP transformers BERT ranking recommendation search systems.
NDCG MRR MAP evaluation frameworks A/B testing offline online metrics.
LLM fine-tuning LoRA QLoRA PEFT RAG retrieval augmented generation reranking.
Learning to rank XGBoost BM25 neural ranking deployed production real users scale.
5 to 9 years experience product companies not consulting TCS Infosys Wipro.
Pune Noida Hyderabad Mumbai Delhi NCR India willing relocate hybrid.
Scrappy product engineering ships fast iterates user feedback open source contributions.
Active job market responds recruiters available short notice period.
"""

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    if title: parts.append(title + " " + title)  # repeat title for weight
    if headline: parts.append(headline)
    if summary: parts.append(summary[:400])

    for job in career[:4]:
        desc = job.get("description", "")
        job_title = job.get("title", "")
        if desc: parts.append(f"{job_title} {desc[:300]}")

    skill_names = [s["name"] for s in skills 
                   if s.get("proficiency") in ("intermediate", "advanced", "expert")]
    if skill_names:
        parts.append(" ".join(skill_names * 2))  # repeat skills for weight

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="./embeddings.npz")
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = []
    opener = gzip.open if args.candidates.endswith(".gz") else open
    with opener(args.candidates, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates")

    print("Building text representations...")
    texts = [build_candidate_text(c) for c in tqdm(candidates)]
    ids = [c["candidate_id"] for c in candidates]

    print("Fitting TF-IDF vectorizer...")
    all_texts = [JD_TEXT] + texts
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    jd_vector = tfidf_matrix[0]
    candidate_matrix = tfidf_matrix[1:]

    print("Computing similarities...")
    similarities = cosine_similarity(candidate_matrix, jd_vector).flatten()

    print(f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    print(f"Saving to {args.out}...")
    
    np.savez_compressed(
        args.out,
        similarities=similarities,
        candidate_ids=np.array(ids)
    )
    print(f"Done! Saved {len(ids):,} scores.")


if __name__ == "__main__":
    main()
