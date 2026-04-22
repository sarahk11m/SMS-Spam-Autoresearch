# SMS Spam AutoResearch — Week 2 Baseline

A minimal, CPU-only AutoResearch project for **STAT 390**.

Research question:

> How much can lightweight text feature engineering improve SMS spam detection on a frozen validation split?

This Week 2 baseline is designed to satisfy the reproducibility gate:
- one end-to-end command
- one fixed validation metric
- one locked test-set plan
- one reproducible README
- one experiment log entry
- one measured runtime

## Week 2 status

- **Current baseline**: TF-IDF (word unigrams + bigrams) + Logistic Regression
- **Primary validation metric**: macro-F1 on the validation split
- **One-line run command**: `python run.py "baseline: tfidf + logistic regression" --baseline`
- **Editable module**: `model.py`
- **Frozen files**: `prepare.py`, `run.py`, `program.md`, and the split logic
- **Locked test-set plan**: the test split is created deterministically in `prepare.py` and is never used in `run.py`
- **Current best result**: see `logs/results.tsv`

## Project structure

```text
sms_spam_autoresearch_week2/
├── README.md
├── program.md
├── prepare.py
├── model.py
├── run.py
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
This repo includes `data/raw/sms_spam_sample.tsv`, a small checked-in SMS corpus so the baseline can run immediately in a fresh clone.

### Planned full dataset
The intended full dataset is the **UCI SMS Spam Collection**, a public binary classification corpus of SMS messages labeled `ham` or `spam`.

When you are ready to switch from the fallback sample to the full corpus, place the UCI file at:

```text
data/raw/SMSSpamCollection
```

`prepare.py` will automatically use it if it exists.

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install scikit-learn matplotlib pandas numpy
```

## How to run

### 1. Baseline run

```bash
python run.py "baseline: tfidf + logistic regression" --baseline
```

Expected behavior:
- loads the checked-in sample corpus (or the full UCI file if present)
- creates deterministic train / validation / test splits
- trains the baseline model from `model.py`
- evaluates **validation macro-F1**
- prints runtime
- appends one row to `logs/results.tsv`

### 2. Plot the experiment history

```bash
python prepare.py
```

This creates:

```text
artifacts/performance.png
```

## Fixed metric and locked test policy

This repo uses:
- **Primary metric**: validation macro-F1
- **Secondary metric**: validation accuracy
- **Split policy**:
  - 60% train
  - 20% validation
  - 20% test
- **Determinism**: all splits use `RANDOM_SEED = 42`
- **Test set access**: `run.py` never evaluates on the test set

## What the agent may change

For this project:
- `model.py` is **EDITABLE**
- `prepare.py`, `run.py`, and `program.md` are **FROZEN**

## First experiment log entry

That entry is stored in:
- `logs/results.tsv`
- `logs/research_log.md`
- `logs/evaluation_board.md`

## Suggested GitHub repo name

```text
sms-spam-autoresearch
```

Suggested GitHub URL template:

```text
https://github.com/YOUR-USERNAME/sms-spam-autoresearch
```
