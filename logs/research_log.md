# Research Log

## 2026-04-29 — Week 3 first dry-run AutoResearch block

### Goal
Launch the first controlled AutoResearch loop by modifying only `model.py` and logging a small set of comparable dry-run experiments.

### Search setup
- **Editable file:** `model.py`
- **Frozen files:** `prepare.py`, `run.py`, `program.md`
- **Primary metric:** validation macro-F1
- **Dataset source used this week:** checked-in fallback SMS sample
- **Why this matters:** Week 3 is about proving that the loop is real, controlled, and interpretable

### Experiment table

| Experiment | Macro-F1 | Accuracy | Status | Training time (sec) |
|---|---:|---:|---|---:|
| baseline: tfidf + logistic regression | 0.9410 | 0.9583 | baseline | 0.0036 |
| dry run 1: unigram tfidf + logistic regression | 0.9410 | 0.9583 | discard | 0.0033 |
| dry run 2: add class_weight balanced | 1.0000 | 1.0000 | discard | 0.0037 |
| dry run 3: char_wb 3-5 grams + logistic regression | 1.0000 | 1.0000 | discard | 0.0063 |
| dry run 4: tfidf + LinearSVC | 1.0000 | 1.0000 | discard | 0.0036 |
| dry run 5: sublinear tf + logistic C=2.0 | 1.0000 | 1.0000 | keep | 0.0039 |

### What I learned
- The loop now works end-to-end with one editable module and a frozen evaluator.
- Small text-feature changes are easy to test quickly.
- On the fallback sample, several variants hit the metric ceiling, so better scores are not automatically strong evidence of real generalization.
- The kept Week 3 candidate was chosen because it stays in the same model family as the baseline while still improving the validation metric.

### What the agent did well
The agent did well at staying inside the editable boundary and producing small, interpretable model changes. It also made it easy to compare runs because every experiment used the same split logic and the same validation metric.

### What the agent did badly
The agent did badly at telling apart a real improvement from a likely small-sample artifact. On a small fallback dataset, several model variants can look equally strong, so raw metric gains can be misleading.

### Reflection
The main success of Week 3 is not the exact score. The success is that the project now has a real, disciplined AutoResearch loop: one editable module, frozen evaluation logic, append-only logs, and interpretable experiment comparisons. The main weakness is that the current fallback sample is small enough to create ceiling effects and ties, so the next step is confirmation on the full UCI SMS Spam Collection.

### Next
- rerun the strongest candidates on the full UCI SMS Spam Collection
- see whether the same ranking holds on a larger corpus
- tighten keep/discard rules for tied or ceiling-level results
