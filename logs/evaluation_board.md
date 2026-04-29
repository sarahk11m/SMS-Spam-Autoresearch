# Evaluation Board

## Frozen search metric
- **Primary metric:** validation macro-F1
- **Secondary metric:** validation accuracy

## Historical baseline
- **Model:** TF-IDF (word unigrams + bigrams) + Logistic Regression
- **Validation macro-F1:** 0.9410
- **Validation accuracy:** 0.9583
- **Training time:** 0.0036 sec

## Current kept candidate
- **Model:** TF-IDF with `sublinear_tf=True` + Logistic Regression with `C=2.0`
- **Validation macro-F1:** 1.0000
- **Validation accuracy:** 1.0000
- **Training time:** 0.0039 sec
- **Important caution:** current kept result is still on the fallback sample only

## Comparison table

| Experiment | Macro-F1 | Accuracy | Status |
|---|---:|---:|---|
| baseline: tfidf + logistic regression | 0.9410 | 0.9583 | baseline |
| dry run 1: unigram tfidf + logistic regression | 0.9410 | 0.9583 | discard |
| dry run 2: add class_weight balanced | 1.0000 | 1.0000 | discard |
| dry run 3: char_wb 3-5 grams + logistic regression | 1.0000 | 1.0000 | discard |
| dry run 4: tfidf + LinearSVC | 1.0000 | 1.0000 | discard |
| dry run 5: sublinear tf + logistic C=2.0 | 1.0000 | 1.0000 | keep |
