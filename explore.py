"""
explore.py
----------
Quick data exploration script. Run this first to understand the dataset.
No ML involved — pure analysis.

Usage:
    python explore.py --candidates ./sample_candidates.json
    python explore.py --candidates ./candidates.jsonl  (for full dataset)
"""

import argparse
import json
import gzip
from collections import Counter
from features import extract_features, is_honeypot

def load(path):
    if path.endswith(".json"):
        return json.load(open(path))
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./sample_candidates.json")
    args = parser.parse_args()

    candidates = load(args.candidates)
    print(f"Loaded {len(candidates):,} candidates\n")

    titles = Counter(c['profile'].get('current_title', 'Unknown') for c in candidates)
    countries = Counter(c['profile'].get('country', 'Unknown') for c in candidates)
    exp_values = [c['profile'].get('years_of_experience', 0) for c in candidates]
    notice_values = [c['redrob_signals'].get('notice_period_days', 0) for c in candidates]
    active_flags = [c['redrob_signals'].get('open_to_work_flag', False) for c in candidates]
    honeypots = [c for c in candidates if is_honeypot(c)]

    print("=== TOP TITLES ===")
    for title, count in titles.most_common(15):
        print(f"  {count:>4}x  {title}")

    print("\n=== COUNTRIES ===")
    for country, count in countries.most_common(10):
        print(f"  {count:>4}x  {country}")

    print(f"\n=== EXPERIENCE ===")
    print(f"  Min: {min(exp_values):.1f} yrs")
    print(f"  Max: {max(exp_values):.1f} yrs")
    print(f"  Avg: {sum(exp_values)/len(exp_values):.1f} yrs")
    in_range = sum(1 for e in exp_values if 5 <= e <= 9)
    print(f"  In 5-9yr range: {in_range}/{len(exp_values)}")

    print(f"\n=== NOTICE PERIODS ===")
    print(f"  ≤30 days: {sum(1 for n in notice_values if n <= 30)}")
    print(f"  31-60d:   {sum(1 for n in notice_values if 30 < n <= 60)}")
    print(f"  61-90d:   {sum(1 for n in notice_values if 60 < n <= 90)}")
    print(f"  >90d:     {sum(1 for n in notice_values if n > 90)}")

    print(f"\n=== AVAILABILITY ===")
    print(f"  Open to work: {sum(active_flags)}/{len(active_flags)}")

    print(f"\n=== HONEYPOTS DETECTED ===")
    print(f"  {len(honeypots)}/{len(candidates)} ({100*len(honeypots)/len(candidates):.1f}%)")
    for h in honeypots[:5]:
        print(f"  - {h['candidate_id']}: {h['profile'].get('current_title')}, {h['profile'].get('years_of_experience')}y")

    # Feature extraction test
    print("\n=== FEATURE EXTRACTION TEST (first 3 candidates) ===")
    for c in candidates[:3]:
        f = extract_features(c)
        print(f"\n{c['candidate_id']} | {c['profile'].get('current_title')} | {c['profile'].get('years_of_experience')}y")
        print(f"  honeypot={f['is_honeypot']} | disqualified={f['is_disqualified_title']} | pure_consulting={f['is_pure_consulting']}")
        print(f"  skill_score={f['required_skill_score']:.3f} | production={f['production_ai_score']:.3f} | location={f['location_fit']:.2f}")
        print(f"  recency={f['recency_score']:.2f} | notice={f['notice_score']:.2f} | response={f['responsiveness_score']:.2f}")

if __name__ == "__main__":
    main()
