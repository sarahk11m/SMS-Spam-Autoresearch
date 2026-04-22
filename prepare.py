"""
FROZEN -- Do not modify this file.
Data loading, deterministic splits, frozen evaluation, logging, and plotting.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_FRACTION = 0.20
VAL_FRACTION_WITHIN_TRAINVAL = 0.25  # gives 60/20/20 overall
RESULTS_FILE = Path("logs/results.tsv")
ARTIFACT_FILE = Path("artifacts/performance.png")
FALLBACK_DATA = Path("data/raw/sms_spam_sample.tsv")
FULL_UCI_DATA = Path("data/raw/SMSSpamCollection")


def _load_full_uci_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            label, text = parts
            rows.append({"label": label.strip(), "text": text.strip()})
    return pd.DataFrame(rows)


def load_dataframe() -> pd.DataFrame:
    """Load the full UCI corpus if present; otherwise use the checked-in fallback sample."""
    if FULL_UCI_DATA.exists():
        df = _load_full_uci_file(FULL_UCI_DATA)
        source = "uci_full"
    elif FALLBACK_DATA.exists():
        df = pd.read_csv(FALLBACK_DATA, sep="\t")
        source = "fallback_sample"
    else:
        raise FileNotFoundError(
            "No dataset found. Expected either data/raw/SMSSpamCollection or data/raw/sms_spam_sample.tsv"
        )
    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    df.attrs["source"] = source
    return df


def load_data():
    """Return deterministic train/validation/test text splits."""
    df = load_dataframe()
    X = df["text"].tolist()
    y = df["label"].tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_FRACTION_WITHIN_TRAINVAL,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    metadata = {
        "dataset_source": df.attrs.get("source", "unknown"),
        "n_total": len(df),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "spam_rate_total": float(np.mean([1 if label == "spam" else 0 for label in y])),
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def evaluate(model, X_val, y_val) -> Tuple[float, float]:
    """Compute frozen validation metrics."""
    y_pred = model.predict(X_val)
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_val, y_pred))
    return macro_f1, accuracy


def log_result(experiment_id: str, val_macro_f1: float, val_accuracy: float,
               status: str, description: str, train_time_sec: float) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "experiment", "val_macro_f1", "val_accuracy",
                "status", "train_time_sec", "description"
            ])
        writer.writerow([
            experiment_id,
            f"{val_macro_f1:.6f}",
            f"{val_accuracy:.6f}",
            status,
            f"{train_time_sec:.4f}",
            description,
        ])


def plot_results(save_path: str | os.PathLike = ARTIFACT_FILE):
    """Plot validation macro-F1 and accuracy over experiments from results.tsv."""
    if not RESULTS_FILE.exists():
        print("No logs/results.tsv found. Run at least one experiment first.")
        return

    experiments, f1s, accs, statuses, descriptions = [], [], [], [], []
    with RESULTS_FILE.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            f1s.append(float(row["val_macro_f1"]))
            accs.append(float(row["val_accuracy"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Macro-F1
    ax1.scatter(range(len(f1s)), f1s, c=colors, s=80, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(f1s)), f1s, "k--", alpha=0.2)
    best_so_far = []
    current_best = -float("inf")
    for score in f1s:
        current_best = max(current_best, score)
        best_so_far.append(current_best)
    ax1.plot(range(len(f1s)), best_so_far, linewidth=2.2, label="Best so far")
    ax1.set_ylabel("Validation macro-F1")
    ax1.set_title("SMS Spam AutoResearch — Metric Trajectory", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # Accuracy
    ax2.scatter(range(len(accs)), accs, c=colors, s=80, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(accs)), accs, "k--", alpha=0.2)
    best_acc = []
    current_best_acc = -float("inf")
    for score in accs:
        current_best_acc = max(current_best_acc, score)
        best_acc.append(current_best_acc)
    ax2.plot(range(len(accs)), best_acc, linewidth=2.2, label="Best so far")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_xlabel("Experiment #")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    short_labels = [d[:28] + ".." if len(d) > 30 else d for d in descriptions]
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
