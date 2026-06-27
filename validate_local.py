"""
validate_local.py
-----------------
Local pre-submission validator.
Run this before uploading to catch format errors.

Usage:
    python validate_local.py --csv ./submission.csv --candidates ./candidates.jsonl
"""

import argparse
import csv
import json
import gzip
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    errors = []
    warnings = []

    # Load valid candidate IDs
    print("Loading candidate IDs...")
    valid_ids = set()
    opener = gzip.open if args.candidates.endswith(".gz") else open
    with opener(args.candidates, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                valid_ids.add(c["candidate_id"])
    print(f"Loaded {len(valid_ids):,} valid IDs")

    # Read CSV
    print("Validating CSV...")
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check columns
        required_cols = {"candidate_id", "rank", "score", "reasoning"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            errors.append(f"Missing columns. Got: {reader.fieldnames}. Need: {required_cols}")

        rows = list(reader)

    # Check row count
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Check ranks
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append(f"Ranks must be 1-100, each exactly once. Got: {sorted(ranks)[:5]}...")

    # Check unique candidate IDs
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate candidate_ids found")

    # Check all IDs exist in dataset
    invalid_ids = [cid for cid in ids if cid not in valid_ids]
    if invalid_ids:
        errors.append(f"Invalid candidate_ids: {invalid_ids[:5]}")

    # Check scores are non-increasing
    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-6:
            errors.append(f"Score at rank {i+1} ({scores[i]:.4f}) > rank {i} ({scores[i-1]:.4f}) — must be non-increasing")
            break

    # Check reasoning quality
    reasonings = [r["reasoning"] for r in rows]
    empty_reasoning = sum(1 for r in reasonings if not r.strip())
    if empty_reasoning > 0:
        warnings.append(f"{empty_reasoning} rows have empty reasoning")

    identical = len(set(reasonings)) < len(reasonings) * 0.5
    if identical:
        warnings.append("More than 50% of reasoning strings are identical — templated reasoning will be penalized")

    # Results
    print()
    if errors:
        print(f"❌ FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print(f"✓ PASSED all format checks")
        print(f"  Rows: {len(rows)}")
        print(f"  Unique IDs: {len(set(ids))}")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  Unique reasoning strings: {len(set(reasonings))}/100")
        if warnings:
            print(f"\n⚠️  Warnings:")
            for w in warnings:
                print(f"   • {w}")

if __name__ == "__main__":
    main()
