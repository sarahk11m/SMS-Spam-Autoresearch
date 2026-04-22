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

## Rules

1. `build_model()` in `model.py` must return an sklearn-compatible estimator or Pipeline.
2. Do not add external datasets or use the final test set during search.
3. Training + validation must finish in under 30 seconds on CPU.
4. The primary metric is **validation macro-F1**.
5. Record each run through `python run.py "<description>"`.
6. If the metric worsens, revert `model.py`.
7. Keep the pipeline lightweight and interpretable.

## Baseline

The baseline is:
- `TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)`
- `LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)`

## Search ideas

- tune n-gram range
- tune `min_df`
- try character n-grams
- try `LinearSVC`
- try class weighting
- try light preprocessing choices
- adjust regularization strength

## What NOT to do

- do not change the metric
- do not change the random seed
- do not evaluate on the test set
- do not rewrite the whole repo
- do not add heavy dependencies
