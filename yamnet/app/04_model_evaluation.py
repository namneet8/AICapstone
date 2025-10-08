"""
04_model_evaluation.py
============================================================
Evaluate the trained audio classification model on the test set.
Generates confidence plots, ROC/PR curves, misclassification analysis,
and inference speed measurement.
============================================================
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

print(list((Path("E:/yamnet/models/finetuned")).glob("*.h5")))

def main():
    print("\n============================================================")
    print("MODEL EVALUATION")
    print("============================================================")

    # Define paths
    model_path = Path(__file__).resolve().parent.parent / "models" / "finetuned" / "final_model.h5"
    # model_path = os.path.join("models", "finetuned", "final_model.h5")
    metadata_path = Path(__file__).resolve().parent.parent / "models" / "finetuned" / "model_metadata.json"
    # metadata_path = os.path.join("models", "finetuned", "model_metadata.json")
    embeddings_path = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.npy"
    # embeddings_path = os.path.join("data", "embeddings", "embeddings.npy")
    labels_path = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "labels.npy"
    # labels_path = os.path.join("data", "embeddings", "labels.npy")

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
    
    # Load model and data
    print("\nLoading model and data...")
    model = tf.keras.models.load_model(model_path)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    class_names = metadata["class_names"]
    num_classes = len(class_names)

    # Recreate test split (same random state as training)
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"\n✓ Model and data loaded!")
    print(f"Test samples: {len(X_test)}")

    # Model predictions
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confidence plots
    plt.figure(figsize=(14, 5))

    max_probs = np.max(y_pred_probs, axis=1)
    plt.subplot(1, 2, 1)
    plt.hist(max_probs, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(np.mean(max_probs), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(max_probs):.3f}")
    plt.title("Prediction Confidence Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Maximum Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    correct = y_pred == y_test
    correct_probs = max_probs[correct]
    incorrect_probs = max_probs[~correct]

    plt.hist(correct_probs, bins=20, alpha=0.7, color="green", label="Correct", edgecolor="black")
    plt.hist(incorrect_probs, bins=20, alpha=0.7, color="red", label="Incorrect", edgecolor="black")
    plt.title("Confidence: Correct vs Incorrect Predictions", fontsize=12, fontweight="bold")
    plt.xlabel("Maximum Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nAverage confidence (correct): {np.mean(correct_probs):.3f}")
    print(f"Average confidence (incorrect): {np.mean(incorrect_probs):.3f}")

    # ROC Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(min(num_classes, 4)):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, color="darkorange", lw=2,
                     label=f"ROC curve (AUC = {roc_auc:.3f})")
        axes[i].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].set_title(f"ROC Curve: {class_names[i]}", fontweight="bold")
        axes[i].legend(loc="lower right")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Precision-Recall Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(min(num_classes, 4)):
        y_binary = (y_test == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_pred_probs[:, i])
        pr_auc = auc(recall, precision)
        axes[i].plot(recall, precision, color="blue", lw=2,
                     label=f"PR curve (AUC = {pr_auc:.3f})")
        axes[i].set_xlabel("Recall")
        axes[i].set_ylabel("Precision")
        axes[i].set_title(f"Precision-Recall: {class_names[i]}", fontweight="bold")
        axes[i].legend(loc="lower left")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Misclassifications
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"\nTotal misclassified: {len(misclassified_idx)} / {len(y_test)} "
          f"({len(misclassified_idx)/len(y_test)*100:.2f}%)")

    print("\nMisclassifications by true class:")
    for i, class_name in enumerate(class_names):
        true_idx = np.where(y_test == i)[0]
        misclass_in_class = np.intersect1d(true_idx, misclassified_idx)
        print(f"  {class_name}: {len(misclass_in_class)} / {len(true_idx)} "
              f"({len(misclass_in_class)/len(true_idx)*100:.2f}%)")

    print("\nMost confused class pairs:")
    cm = confusion_matrix(y_test, y_pred)
    confused_pairs = [
        (i, j, cm[i, j]) for i in range(num_classes) for j in range(num_classes)
        if i != j and cm[i, j] > 0
    ]
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for t, p, c in confused_pairs[:5]:
        print(f"  {class_names[t]} → {class_names[p]}: {c} samples")

    # Sample predictions visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    correct_idx = np.where(y_pred == y_test)[0]

    for i in range(min(3, len(correct_idx))):
        idx = correct_idx[i]
        ax = axes[0, i]
        probs = y_pred_probs[idx]
        ax.bar(range(num_classes), probs, color="green", alpha=0.7)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"✓ True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
                     fontsize=10, fontweight="bold", color="green")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(min(3, len(misclassified_idx))):
        idx = misclassified_idx[i]
        ax = axes[1, i]
        probs = y_pred_probs[idx]
        colors = [
            "red" if j == y_pred[idx] else
            "orange" if j == y_test[idx] else
            "gray"
            for j in range(num_classes)
        ]
        ax.bar(range(num_classes), probs, color=colors, alpha=0.7)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"✗ True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
                     fontsize=10, fontweight="bold", color="red")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Sample Predictions: Correct (Top) vs Incorrect (Bottom)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Inference speed test
    print("\nMeasuring inference speed...")
    num_trials = 100
    times = []
    for _ in range(num_trials):
        sample = X_test[np.random.randint(0, len(X_test))].reshape(1, -1)
        start = time.time()
        _ = model.predict(sample, verbose=0)
        times.append(time.time() - start)

    print(f"Average inference time: {np.mean(times)*1000:.2f} ms/sample")
    print("============================================================\n")


if __name__ == "__main__":
    main()
