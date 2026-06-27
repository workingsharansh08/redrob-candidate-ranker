"""
embed.py - HYBRID VERSION (TF-IDF + Word2Vec)
- Trains Word2Vec directly on your 100K candidate corpus (no download needed)
- Combines TF-IDF similarity + Word2Vec semantic similarity
- Better semantic understanding than TF-IDF alone
- Still fully offline, no internet required
"""

import argparse
import json
import gzip
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re

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
machine learning deep learning pytorch tensorflow huggingface transformers
recommendation system collaborative filtering content based embedding similarity
vector search approximate nearest neighbor index retrieval pipeline
"""

def tokenize(text):
    """Simple tokenizer — lowercase, split on non-alphanumeric"""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    if title: parts.append(title + " " + title)
    if headline: parts.append(headline)
    if summary: parts.append(summary[:400])

    for job in career[:4]:
        desc = job.get("description", "")
        job_title = job.get("title", "")
        if desc: parts.append(f"{job_title} {desc[:300]}")

    skill_names = [s["name"] for s in skills
                   if s.get("proficiency") in ("intermediate", "advanced", "expert")]
    if skill_names:
        parts.append(" ".join(skill_names * 2))

    return " ".join(parts)


def get_w2v_vector(tokens, model, size=100):
    """Average Word2Vec vectors for a list of tokens"""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(size)
    return np.mean(vecs, axis=0)


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

    # ── TF-IDF similarity ──────────────────────────────────────────────────
    print("Computing TF-IDF similarities...")
    all_texts = [JD_TEXT] + texts
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vector = tfidf_matrix[0]
    candidate_matrix = tfidf_matrix[1:]
    tfidf_sims = cosine_similarity(candidate_matrix, jd_vector).flatten()
    print(f"TF-IDF similarity range: [{tfidf_sims.min():.3f}, {tfidf_sims.max():.3f}]")

    # ── Word2Vec training ──────────────────────────────────────────────────
    print("Tokenizing corpus for Word2Vec...")
    tokenized = [tokenize(t) for t in tqdm(texts)]
    jd_tokens = tokenize(JD_TEXT)

    print("Training Word2Vec on candidate corpus (this takes ~2-3 min)...")
    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=150,
        window=6,
        min_count=3,
        workers=4,
        epochs=5,
        sg=1  # skip-gram — better for rare words
    )
    print(f"Word2Vec vocab size: {len(w2v_model.wv):,}")

    print("Computing Word2Vec similarities...")
    jd_w2v = get_w2v_vector(jd_tokens, w2v_model, size=150)
    w2v_sims = np.array([
        cosine_similarity(
            get_w2v_vector(tokens, w2v_model, size=150).reshape(1, -1),
            jd_w2v.reshape(1, -1)
        )[0][0]
        for tokens in tqdm(tokenized)
    ])
    print(f"Word2Vec similarity range: [{w2v_sims.min():.3f}, {w2v_sims.max():.3f}]")

    # ── Combine: 60% TF-IDF + 40% Word2Vec ────────────────────────────────
    print("Combining similarities...")

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    tfidf_norm = normalize(tfidf_sims)
    w2v_norm = normalize(w2v_sims)
    combined = 0.60 * tfidf_norm + 0.40 * w2v_norm

    print(f"Combined similarity range: [{combined.min():.3f}, {combined.max():.3f}]")
    print(f"Saving to {args.out}...")

    np.savez_compressed(
        args.out,
        similarities=combined,
        tfidf_similarities=tfidf_norm,
        w2v_similarities=w2v_norm,
        candidate_ids=np.array(ids)
    )
    print(f"Done! Saved {len(ids):,} scores.")
    print("Hybrid TF-IDF + Word2Vec embeddings ready.")


if __name__ == "__main__":
    main()
