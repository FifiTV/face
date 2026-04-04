"""
Biometric metrics: FAR, FRR, accuracy, ROC curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_far_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Compute FAR, FRR and accuracy at a given threshold.

    Args:
        scores: cosine similarity scores, shape (N,)
        labels: 1 = genuine pair, 0 = impostor pair, shape (N,)
        threshold: decision boundary
    Returns:
        (FAR, FRR, accuracy)
    """
    decisions = (scores >= threshold).astype(int)
    # FAR: impostors accepted / total impostors
    impostors = labels == 0
    genuines = labels == 1
    far = np.sum((decisions == 1) & impostors) / max(np.sum(impostors), 1)
    frr = np.sum((decisions == 0) & genuines) / max(np.sum(genuines), 1)
    acc = np.mean(decisions == labels)
    return float(far), float(frr), float(acc)


def compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    Returns (fpr, tpr, thresholds).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    return fpr, tpr, thresholds


def eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and corresponding threshold."""
    frr = 1 - tpr
    idx = np.argmin(np.abs(fpr - frr))
    return float((fpr[idx] + frr[idx]) / 2), float(thresholds[idx])


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, ax: plt.Axes | None = None, label: str = "") -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (1-FRR)")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_far_frr_vs_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    fars, frrs = [], []
    for t in thresholds:
        f, r, _ = compute_far_frr(scores, labels, t)
        fars.append(f)
        frrs.append(r)
    ax.plot(thresholds, fars, label="FAR")
    ax.plot(thresholds, frrs, label="FRR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR vs Threshold")
    ax.legend()
    return ax
