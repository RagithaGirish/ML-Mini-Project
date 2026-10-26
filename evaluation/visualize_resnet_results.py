"""Create plots and a per-class metrics table from saved ResNet results.

Run from the project root after baseline_resnet.py has finished:
    python evaluation/visualize_resnet_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """Plot loss, validation accuracy and validation macro-F1 for every epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", color="#d95f02")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["validation_accuracy"], marker="o", label="Accuracy")
    axes[1].plot(history["epoch"], history["validation_macro_f1"], marker="s", label="Macro F1")
    axes[1].set(title="Validation Performance", xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("ResNet-18 Training History")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    """Save a labeled confusion-matrix figure for the held-out test set."""
    fig, axis = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=30, values_format="d")
    axis.set_title("ResNet-18 Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_class_table(report: dict) -> pd.DataFrame:
    """Convert sklearn's JSON classification report to a report-friendly CSV."""
    rows = []
    for class_name, values in report.items():
        if class_name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append({
            "class": class_name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved ResNet-18 training results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/baseline_resnet"))
    args = parser.parse_args()

    history_path = args.results_dir / "history.csv"
    metrics_path = args.results_dir / "test_metrics.json"
    if not history_path.exists() or not metrics_path.exists():
        parser.error("history.csv and test_metrics.json must exist. Run baseline_resnet.py first.")

    history = pd.read_csv(history_path)
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    plot_training_history(history, args.results_dir / "training_history.png")
    classes = list(metrics["classification_report"].keys())[:5]
    plot_confusion_matrix(metrics["confusion_matrix"], classes, args.results_dir / "confusion_matrix.png")
    per_class_table(metrics["classification_report"]).to_csv(args.results_dir / "per_class_metrics.csv", index=False)

    best_epoch = history.loc[history["validation_macro_f1"].idxmax()]
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Best validation macro-F1: {best_epoch['validation_macro_f1']:.4f} (epoch {int(best_epoch['epoch'])})")
    print(f"Saved plots and per-class metrics to {args.results_dir}")


if __name__ == "__main__":
    main()
