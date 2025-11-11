"""
Script to analyze SciBERT Multimodal model training results and test predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Paths
MODEL_DIR = Path("/Users/jasonyan/Desktop/CS 230/NoveltyRank/models/scibert_multimodal")
TRAINING_HISTORY = MODEL_DIR / "training_history.csv"
TEST_PREDICTIONS = MODEL_DIR / "test_predictions.csv"
OUTPUT_DIR = MODEL_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_training_history():
    """Analyze and visualize training history."""
    print("=" * 80)
    print("TRAINING HISTORY ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(TRAINING_HISTORY)
    print("\nTraining History:")
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST EPOCH SUMMARY")
    print("=" * 80)
    best_epoch = df.loc[df["test_f1"].idxmax()]
    print(f"\nBest Epoch: {int(best_epoch['epoch'])}")
    print(f"Train Loss: {best_epoch['train_loss']:.4f}")
    print(f"Train Accuracy: {best_epoch['train_accuracy']:.4f}")
    print(f"Test Loss: {best_epoch['test_loss']:.4f}")
    print(f"Test Accuracy: {best_epoch['test_accuracy']:.4f}")
    print(f"Test Precision: {best_epoch['test_precision']:.4f}")
    print(f"Test Recall: {best_epoch['test_recall']:.4f}")
    print(f"Test F1: {best_epoch['test_f1']:.4f}")
    print(f"Test AUC-ROC: {best_epoch['test_auc_roc']:.4f}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(
        df["epoch"], df["train_loss"], "o-", label="Train Loss", linewidth=2
    )
    axes[0, 0].plot(df["epoch"], df["test_loss"], "s-", label="Test Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("Training and Test Loss", fontsize=14, fontweight="bold")
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(
        df["epoch"], df["train_accuracy"], "o-", label="Train Accuracy", linewidth=2
    )
    axes[0, 1].plot(
        df["epoch"], df["test_accuracy"], "s-", label="Test Accuracy", linewidth=2
    )
    axes[0, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 1].set_ylabel("Accuracy", fontsize=12)
    axes[0, 1].set_title("Training and Test Accuracy", fontsize=14, fontweight="bold")
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.7, 1.0])

    # Precision, Recall, F1
    axes[1, 0].plot(
        df["epoch"], df["test_precision"], "o-", label="Precision", linewidth=2
    )
    axes[1, 0].plot(df["epoch"], df["test_recall"], "s-", label="Recall", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["test_f1"], "^-", label="F1 Score", linewidth=2)
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 0].set_ylabel("Score", fontsize=12)
    axes[1, 0].set_title(
        "Test Metrics (Precision, Recall, F1)", fontsize=14, fontweight="bold"
    )
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.7, 1.0])

    # AUC-ROC
    axes[1, 1].plot(
        df["epoch"],
        df["test_auc_roc"],
        "o-",
        label="AUC-ROC",
        linewidth=2,
        color="purple",
    )
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_ylabel("AUC-ROC", fontsize=12)
    axes[1, 1].set_title("Test AUC-ROC", fontsize=14, fontweight="bold")
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.96, 0.98])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved training history plot to {OUTPUT_DIR / 'training_history.png'}")
    plt.close()

    return best_epoch


def analyze_test_predictions():
    """Analyze test predictions in detail."""
    print("\n" + "=" * 80)
    print("TEST PREDICTIONS ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(TEST_PREDICTIONS)

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    y_prob = df["probability_positive"].values

    # Basic statistics
    print(f"\nTotal test samples: {len(df)}")
    print(
        f"Positive class (accepted): {(y_true == 1).sum()} ({100 * (y_true == 1).sum() / len(df):.2f}%)"
    )
    print(
        f"Negative class (rejected): {(y_true == 0).sum()} ({100 * (y_true == 0).sum() / len(df):.2f}%)"
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print("\n                 Predicted")
    print("                 Rejected  Accepted")
    print(f"Actual Rejected  {cm[0, 0]:>8}  {cm[0, 1]:>8}")
    print(f"       Accepted  {cm[1, 0]:>8}  {cm[1, 1]:>8}")

    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")

    # Classification Report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(
        "\n"
        + classification_report(
            y_true, y_pred, target_names=["Rejected", "Accepted"], digits=4
        )
    )

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    # Create visualizations
    fig = plt.figure(figsize=(18, 12))

    # Confusion Matrix Heatmap
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Rejected", "Accepted"],
        yticklabels=["Rejected", "Accepted"],
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Normalized Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar=True,
        xticklabels=["Rejected", "Accepted"],
        yticklabels=["Rejected", "Accepted"],
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")

    # ROC Curve
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(
        recall_curve,
        precision_curve,
        color="blue",
        lw=2,
        label=f"PR curve (AP = {avg_precision:.4f})",
    )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)

    # Probability Distribution
    ax5 = plt.subplot(2, 3, 5)
    plt.hist(
        y_prob[y_true == 0],
        bins=50,
        alpha=0.5,
        label="Rejected (True)",
        color="red",
        density=True,
    )
    plt.hist(
        y_prob[y_true == 1],
        bins=50,
        alpha=0.5,
        label="Accepted (True)",
        color="green",
        density=True,
    )
    plt.xlabel("Predicted Probability (Positive Class)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Error Analysis - Misclassified samples
    ax6 = plt.subplot(2, 3, 6)
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)

    fp_probs = y_prob[false_positives]
    fn_probs = y_prob[false_negatives]

    plt.hist(
        fp_probs,
        bins=30,
        alpha=0.7,
        label=f"False Positives (n={len(fp_probs)})",
        color="orange",
    )
    plt.hist(
        fn_probs,
        bins=30,
        alpha=0.7,
        label=f"False Negatives (n={len(fn_probs)})",
        color="red",
    )
    plt.xlabel("Predicted Probability (Positive Class)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Misclassified Samples Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "test_predictions_analysis.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"\n✓ Saved test predictions analysis to {OUTPUT_DIR / 'test_predictions_analysis.png'}"
    )
    plt.close()

    # Error Analysis Details
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    print(f"\nFalse Positives: {fp} (predicted accepted but actually rejected)")
    print(f"False Negatives: {fn} (predicted rejected but actually accepted)")

    if len(fp_probs) > 0:
        print(f"\nFalse Positive probabilities:")
        print(f"  Mean: {fp_probs.mean():.4f}")
        print(f"  Median: {np.median(fp_probs):.4f}")
        print(f"  Min: {fp_probs.min():.4f}")
        print(f"  Max: {fp_probs.max():.4f}")

    if len(fn_probs) > 0:
        print(f"\nFalse Negative probabilities:")
        print(f"  Mean: {fn_probs.mean():.4f}")
        print(f"  Median: {np.median(fn_probs):.4f}")
        print(f"  Min: {fn_probs.min():.4f}")
        print(f"  Max: {fn_probs.max():.4f}")

    # Save detailed metrics to file
    metrics_summary = {
        "Total Samples": len(df),
        "Positive Class Count": int((y_true == 1).sum()),
        "Negative Class Count": int((y_true == 0).sum()),
        "True Positives": int(tp),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1 Score": float(f1),
        "AUC-ROC": float(roc_auc),
        "Average Precision": float(avg_precision),
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print(f"\n✓ Saved metrics summary to {OUTPUT_DIR / 'metrics_summary.csv'}")


def main():
    print("\n" + "=" * 80)
    print("SCIBERT MULTIMODAL MODEL - RESULTS ANALYSIS")
    print("=" * 80)

    # Analyze training history
    best_epoch = analyze_training_history()

    # Analyze test predictions
    analyze_test_predictions()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - training_history.png")
    print(f"  - test_predictions_analysis.png")
    print(f"  - metrics_summary.csv")


if __name__ == "__main__":
    main()
