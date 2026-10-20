"""Classification metrics.

The classes are imbalanced (after deduplication, dog has ~3.6x as many training images as elephant),
so accuracy alone is misleading; macro-F1, where every class counts equally, is the primary
model-selection metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 5) -> list[dict]:
    """The k most frequent (true -> predicted) mistakes."""
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:k]
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, cm.shape)
        if off_diagonal[i, j] == 0:
            break
        pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(off_diagonal[i, j])})
    return pairs
