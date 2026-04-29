# AutoResearch Agent Instructions

## Objective

Maximize **validation macro-F1** for SMS spam detection on the frozen validation split.

## Editable boundary

You may **ONLY** modify `model.py`.

## Frozen files

Do **NOT** modify:
- `prepare.py`
- `run.py`
- `program.md`
- anything in `logs/`
- the train / validation / test split logic

## Required behavior

1. `build_model()` in `model.py` must return an sklearn-compatible estimator or Pipeline.
2. Keep the search lightweight and CPU-friendly.
3. Training + validation should finish in under 30 seconds.
4. Record each run through `python run.py "<description>"`.
5. The primary metric is **validation macro-F1**.
6. Do not use the test set during search.
7. Prefer small, interpretable changes over broad rewrites.

## Keep / discard / crash rules

- **KEEP** if validation macro-F1 improves in a way that is still interpretable and simple enough to justify.
- **DISCARD** if the metric worsens, ties without a compelling simplicity advantage, or looks too likely to be a small-sample artifact.
- **CRASH** if code fails, outputs break, dependencies are missing, or evaluation cannot complete.

## Historical baseline

The historical baseline used for Week 2 was:
- `TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=False)`
- `LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)`

## Current kept candidate

The current kept candidate in `model.py` is:
- `TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=True)`
- `LogisticRegression(max_iter=1000, solver="liblinear", C=2.0, random_state=42)`

## Week 3 dry-run search ideas

- unigram vs unigram+bigrams
- class weighting
- character n-grams
- `LinearSVC`
- small TF-IDF weighting changes
- light regularization changes

## What NOT to do

- do not change the metric
- do not change the random seed
- do not change the split policy
- do not evaluate on the test set
- do not rewrite the repo
- do not add heavy dependencies
