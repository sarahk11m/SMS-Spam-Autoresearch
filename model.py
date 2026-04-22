"""
EDITABLE -- The agent modifies only this file.
Define the model pipeline for SMS spam classification.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model():
    """Return an sklearn-compatible baseline pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )),
    ])
