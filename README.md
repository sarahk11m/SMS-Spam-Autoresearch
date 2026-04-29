# SMS Spam AutoResearch — Week 3

A minimal, CPU-only AutoResearch project for **STAT 390**.

## Research question

> How much can lightweight text feature engineering improve SMS spam detection on a frozen validation split?

## Week 3 status

This repo now includes a real first AutoResearch loop:

- **Editable file:** `model.py`
- **Frozen files:** `prepare.py`, `run.py`, `program.md`, and the fixed split logic
- **Primary metric:** validation macro-F1
- **Historical baseline:** TF-IDF (word unigrams + bigrams) + Logistic Regression
- **Historical baseline result:** 0.9410 validation macro-F1 on the checked-in fallback sample
- **Current kept candidate in `model.py`:** TF-IDF with `sublinear_tf=True` + Logistic Regression with `C=2.0`
- **Current kept candidate result:** 1.0000 validation macro-F1 on the checked-in fallback sample
- **Important caution:** the checked-in fallback sample is small, so Week 3 improvements should be treated as dry-run signals rather than confirmed gains

## What changed in Week 3

Week 2 established the frozen evaluation pipeline.  
Week 3 adds:

- a first real `program.md`
- 5 dry-run experiments logged in `logs/results.tsv`
- a written research reflection in `logs/research_log.md`
- a failure-mode list in `logs/failure_log.md`
- an updated metric trajectory plot in `artifacts/performance.png`

## Project structure

```text
sms_spam_week3_work/
├── README.md
├── program.md
├── prepare.py                # FROZEN: data load, deterministic split, evaluation, plotting
├── model.py                  # EDITABLE: current kept candidate
├── run.py                    # FROZEN: one experiment runner
├── .gitignore
├── data/
│   └── raw/
│       └── sms_spam_sample.tsv
├── logs/
│   ├── results.tsv
│   ├── research_log.md
│   ├── failure_log.md
│   └── evaluation_board.md
└── artifacts/
    └── performance.png
```

## Data

### Checked-in fallback sample
This repo includes `data/raw/sms_spam_sample.tsv`, a small checked-in SMS corpus so the loop can run immediately in a fresh clone.

### Planned full dataset
The intended full dataset is the **UCI SMS Spam Collection**, a public binary classification corpus of SMS messages labeled `ham` or `spam`.

If the UCI file is placed at:

```text
data/raw/SMSSpamCollection
```

then `prepare.py` will automatically use it instead of the fallback sample.

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install scikit-learn matplotlib pandas numpy
```

## How to run

### 1. Run the current candidate in `model.py`

```bash
python run.py "rerun current candidate"
```

This will:
- load the fallback sample or the full UCI file if present
- apply the deterministic 60/20/20 split
- train the current model from `model.py`
- evaluate validation macro-F1 and validation accuracy
- append the result to `logs/results.tsv`

### 2. Plot the experiment history

```bash
python prepare.py
```

This updates:

```text
artifacts/performance.png
```

## Fixed metric and locked test policy

This repo uses:
- **Primary metric:** validation macro-F1
- **Secondary metric:** validation accuracy
- **Split policy:**
  - 60% train
  - 20% validation
  - 20% test
- **Determinism:** `RANDOM_SEED = 42`
- **Test set access:** `run.py` loads the test split but does not evaluate on it during search

## Week 3 dry-run experiment summary

| Experiment | Macro-F1 | Accuracy | Status |
|---|---:|---:|---|
| baseline: tfidf + logistic regression | 0.9410 | 0.9583 | baseline |
| dry run 1: unigram tfidf + logistic regression | 0.9410 | 0.9583 | discard |
| dry run 2: add class_weight balanced | 1.0000 | 1.0000 | discard |
| dry run 3: char_wb 3-5 grams + logistic regression | 1.0000 | 1.0000 | discard |
| dry run 4: tfidf + LinearSVC | 1.0000 | 1.0000 | discard |
| dry run 5: sublinear tf + logistic C=2.0 | 1.0000 | 1.0000 | keep |

## Interpretation of Week 3

The loop is real and reproducible, but the checked-in fallback sample is small enough that several variants hit the metric ceiling. That means Week 3 is a proof of controlled iteration, not a strong claim that the kept candidate is definitively better on the full problem.

## Suggested next step

Repeat the strongest candidates on the full UCI SMS Spam Collection and see whether the same ranking holds once the evaluator is applied to a larger corpus.
