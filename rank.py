"""
rank.py - Main ranking script
Run AFTER embed.py finishes.

Usage:
    python rank.py --candidates "candidates (2).jsonl" --embeddings ./embeddings.npz --out ./submission.csv
"""

import argparse
import json
import gzip
import csv
import time
import numpy as np
from tqdm import tqdm

from features import extract_features
from score import rank_candidates
from reasoning import generate_reasoning


def load_candidates(path):
    candidates = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates[c["candidate_id"]] = c
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", default="./submission.csv")
    args = parser.parse_args()

    t0 = time.time()

    # Load precomputed similarities
    print("[1/5] Loading precomputed similarities...")
    data = np.load(args.embeddings, allow_pickle=True)
    similarities = data["similarities"]
    candidate_ids = data["candidate_ids"]
    print(f"      Loaded {len(candidate_ids):,} candidates in {time.time()-t0:.1f}s")

    # Normalize to [0,1]
    sim_min, sim_max = similarities.min(), similarities.max()
    similarities_norm = (similarities - sim_min) / (sim_max - sim_min + 1e-9)

    # Load candidates
    print("[2/5] Loading candidate profiles...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # Extract features
    print("[3/5] Extracting features...")
    all_features = []
    all_sims = []
    id_to_idx = {str(cid): i for i, cid in enumerate(candidate_ids)}

    for cid in tqdm(candidate_ids, desc="Features"):
        candidate = candidates.get(str(cid))
        if candidate is None:
            continue
        features = extract_features(candidate)
        idx = id_to_idx[str(cid)]
        all_features.append(features)
        all_sims.append(float(similarities_norm[idx]))

    print(f"      Done in {time.time()-t0:.1f}s")

    # Score and rank
    print("[4/5] Scoring and ranking...")
    ranked = rank_candidates(all_features, all_sims)
    top_100 = ranked[:100]

    # Write CSV
    print(f"[5/5] Writing {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (cid, score, features) in enumerate(top_100, start=1):
            candidate = candidates[str(cid)]
            reasoning = generate_reasoning(candidate, features, score, rank_pos)
            writer.writerow([cid, rank_pos, f"{score:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"✓ Submission written to {args.out}")
    print(f"\nTop 10 candidates:")
    print(f"{'Rank':<6} {'ID':<15} {'Score':<10} {'Title':<30} {'YoE'}")
    print("-" * 70)
    for rank_pos, (cid, score, features) in enumerate(top_100[:10], start=1):
        c = candidates[str(cid)]
        title = c['profile'].get('current_title', 'Unknown')[:28]
        yoe = c['profile'].get('years_of_experience', 0)
        print(f"{rank_pos:<6} {str(cid):<15} {score:<10.4f} {title:<30} {yoe:.1f}y")

    if elapsed > 240:
        print(f"\n⚠️  WARNING: {elapsed:.0f}s — close to 5-min limit!")
    else:
        print(f"\n✓ Runtime {elapsed:.1f}s / 300s limit — safe")


if __name__ == "__main__":
    main()
