# Research Log

## 2026-04-22 — Week 2 baseline established

### Goal
Freeze the evaluation instrument for the SMS spam project and produce a reproducible baseline.

### What I ran
`python run.py "baseline: tfidf + logistic regression" --baseline`

### Result
- validation macro-F1: **0.9410**
- validation accuracy: **0.9583**
- measured training time: **0.0138 sec**
- dataset source used in this repo: **checked-in fallback sample**

### Why this matters
This gives the project a stable baseline before any agent-driven search begins.

### Next
- verify the same pipeline on the full UCI SMS Spam Collection
- begin Week 3 dry-run experiments by changing only `model.py`
