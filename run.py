"""
Run one experiment: build model, train, evaluate on validation, and log result.

Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard
"""
from __future__ import annotations

import subprocess
import sys
import time

from prepare import evaluate, load_data, log_result


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main() -> None:
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)

    description = " ".join(description_parts) if description_parts else "experiment"

    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_data()
    print(f"Dataset source: {metadata['dataset_source']}")
    print(
        f"Data: {metadata['n_train']} train, {metadata['n_val']} val, {metadata['n_test']} test "
        f"(total={metadata['n_total']})"
    )

    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.4f}s")

    val_macro_f1, val_accuracy = evaluate(model, X_val, y_val)
    print(f"val_macro_f1: {val_macro_f1:.6f}")
    print(f"val_accuracy: {val_accuracy:.6f}")

    experiment_id = get_git_hash()
    log_result(
        experiment_id=experiment_id,
        val_macro_f1=val_macro_f1,
        val_accuracy=val_accuracy,
        status=status,
        description=description,
        train_time_sec=train_time,
    )
    print(f"Result logged to logs/results.tsv (status={status})")
    print("Locked test-set policy: X_test and y_test were loaded but not evaluated in this script.")


if __name__ == "__main__":
    main()
