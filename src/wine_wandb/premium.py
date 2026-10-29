"""The premium-wine task: the label, the classifier pipeline, and its evaluation metrics.

The regression target `quality` is turned into a binary label (quality >= 7). The model's
predicted class (`predict`) is used as the decision.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREMIUM_MIN_QUALITY = 7


def premium(quality):
    return (np.asarray(quality) >= PREMIUM_MIN_QUALITY).astype(int)


def build_classifier(seed: int, n_estimators: int = 300) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ]
    )


def classification_metrics(y_true, y_pred, score) -> dict:
    """Counts and metrics of the predicted classes, plus the threshold-free AUCs of the scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "accuracy": float((tp + tn) / len(y_true)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
