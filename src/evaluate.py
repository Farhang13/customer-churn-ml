import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

def evaluate_at_threshold(y_true, probs, threshold: float = 0.5):
    """
    Evaluate binary classifier probabilities at a given threshold.

    y_true: array-like of 0/1 labels
    probs: array-like of predicted probabilities for class 1
    threshold: probability cutoff to predict churn (1)
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # aka TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Useful business-style rates
    churn_catch_rate = recall              # how many churners you catch
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0  # FPR

    print(f"\n=== Evaluation @ threshold = {threshold:.2f} ===")
    print("Confusion matrix [ [TN FP]\n                 [FN TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Key metrics:")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision (PPV):    {precision:.4f}")
    print(f"  Recall (TPR):       {recall:.4f}  (churners caught)")
    print(f"  F1-score:           {f1:.4f}")
    print(f"  False alarm rate:   {false_alarm_rate:.4f}  (non-churners flagged)")

    return {
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }