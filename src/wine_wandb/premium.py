"""The premium-wine task: label, classifier, and a cost-based decision threshold.

Shared by `train_classifier.py` (tracked run), `evaluate.py`, and `decision.py` (figures).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREMIUM_MIN_QUALITY = 7
COST_FALSE_POSITIVE = 1.0  # a standard wine sold as premium
COST_FALSE_NEGATIVE = 5.0  # a premium wine sold as standard
THRESHOLD_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def premium(quality):
    return (np.asarray(quality) >= PREMIUM_MIN_QUALITY).astype(int)


def build_classifier(seed: int, n_estimators: int = 300) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ]
    )


def counts(y_true, score, tau):
    tn, fp, fn, tp = confusion_matrix(y_true, (score >= tau).astype(int), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def cost(fp, fn, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    return c_fp * fp + c_fn * fn


def summarize(y_true, score, tau):
    tn, fp, fn, tp = counts(y_true, score, tau)
    return {
        "tau": float(tau),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / len(y_true),
        "cost": cost(fp, fn),
    }


def choose_threshold(y_true, score, grid):
    """Threshold minimizing the cost; ties resolved by the median of the minimizers."""
    costs = np.array([cost(*counts(y_true, score, t)[1:3]) for t in grid])
    return float(np.median(grid[costs == costs.min()])), costs


def out_of_fold_scores(pipeline, X, y, folds: int, seed: int):
    """Probabilities of the positive class for every training wine, from models that never saw it."""
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(pipeline, X, y, cv=splitter, method="predict_proba")[:, 1]
