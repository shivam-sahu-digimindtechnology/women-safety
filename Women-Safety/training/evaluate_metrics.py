"""
Evaluation Metrics & Diagnostics Module
=======================================
Computes benchmark metrics for Women Safety AI detection models:
- Accuracy, Precision, Recall, F1-Score
- False Positive Rate (FPR) & False Negative Rate (FNR)
- Detection Latency & FPS throughput
- Confusion Matrix display

Used for model verification and regression testing.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_binary_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))

    total = len(y_t)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        },
        "total_samples": total
    }


def print_metrics_report(title: str, metrics: Dict[str, Any]) -> None:
    """Pretty prints evaluation metrics."""
    print(f"\n==========================================")
    print(f" METRICS REPORT: {title.upper()}")
    print(f"==========================================")
    print(f" Total Samples          : {metrics['total_samples']}")
    print(f" Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print(f" Precision              : {metrics['precision'] * 100:.2f}%")
    print(f" Recall (Sensitivity)   : {metrics['recall'] * 100:.2f}%")
    print(f" F1-Score               : {metrics['f1_score']:.4f}")
    print(f" False Positive Rate    : {metrics['false_positive_rate'] * 100:.2f}%")
    print(f" False Negative Rate    : {metrics['false_negative_rate'] * 100:.2f}%")
    cm = metrics["confusion_matrix"]
    print(f" Confusion Matrix       : TP={cm['TP']} | FP={cm['FP']} | FN={cm['FN']} | TN={cm['TN']}")
    print(f"==========================================\n")


if __name__ == "__main__":
    # Diagnostic self-test
    mock_true = [0, 0, 0, 1, 1, 1, 0, 1, 0, 0]
    mock_pred = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    res = compute_binary_metrics(mock_true, mock_pred)
    print_metrics_report("Mock Violence Classifier Test", res)
