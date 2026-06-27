# Redrob Intelligent Candidate Ranker

**Challenge:** India RUNS — Data & AI Challenge (Hack2Skill × Redrob)  
**Task:** Rank top 100 candidates from 100K pool for a Senior AI Engineer JD

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
   [embed.py] ──── Sentence-Transformer (all-MiniLM-L6-v2)
        │           Encodes candidate text + JD text
        ▼
  embeddings.npz  ──────────────────────────────────┐
        │                                            │
   [rank.py]                                         │
        │                                            │
        ├── Load embeddings (numpy, fast)            │
        ├── Cosine similarity vs JD embedding ◄──────┘
        ├── [features.py] — extract 20+ structured features
        ├── [score.py] — weighted scoring formula
        ├── [reasoning.py] — candidate-specific reasoning
        └── submission.csv (top 100, ranked)
```

## How to Run

### Step 1: Install dependencies (once)
```bash
pip install -r requirements.txt
```

### Step 2: Precompute embeddings (run once, ~30-40 min on CPU)
```bash
python embed.py --candidates ./candidates.jsonl --out ./embeddings.npz
```

### Step 3: Generate submission (must complete < 5 min)
```bash
python rank.py --candidates ./candidates.jsonl --embeddings ./embeddings.npz --out ./submission.csv
```

### Step 4: Validate format
```bash
python validate_submission.py ./submission.csv
```

### Step 5: Explore data (optional)
```bash
python explore.py --candidates ./sample_candidates.json
```

## Scoring Formula

```
Final Score = weighted sum of:
  - Semantic similarity (embedding cosine sim vs JD): 30%
  - Required skill match score:                       20%
  - Production AI deployment evidence:                15%
  - Behavioral availability score:                    15%
  - Experience fit (5-9 yrs ideal):                    8%
  - Product company ratio:                             5%
  - Location fit (Pune/Noida/Hyd/Mumbai/Delhi):        4%
  - Education tier:                                    2%
  - GitHub activity:                                   1%
```

## Hard Disqualifications
- Honeypot profiles (impossible timelines) → score ≈ 0
- Disqualified titles (Marketing Manager, Accountant etc.) with no ML skills → score ≈ 0
- Pure consulting career (entire career at TCS/Wipro/Infosys) with no ML skills → score ≈ 0

## Design Decisions

**Why sentence-transformers over keyword matching?**  
The JD explicitly says "A Tier 5 candidate may not use the words RAG or Pinecone but if their career history shows they built a recommendation system at a product company, they're a fit." Semantic embeddings capture this; keyword matching does not.

**Why precompute embeddings?**  
Embedding 100K candidates in real-time would exceed the 5-minute CPU constraint. Precomputation decouples the expensive step (encoding) from the constrained step (ranking).

**Why behavioral signals as a multiplier?**  
The JD and signals doc both say: "A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available." Availability is a hard real-world constraint, not just a nice-to-have.

**Why hard filters before scoring?**  
The sample submission included HR Managers and Accountants in top 10. The evaluation is NDCG@10 weighted 50%. One wrong candidate in the top 10 costs ~5% of your total score.
