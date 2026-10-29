import numpy as np
import pandas as pd

from wine_wandb import premium


def test_premium_label_cutoff():
    assert premium.premium([5, 6, 7, 8]).tolist() == [0, 0, 1, 1]


def test_classification_metrics_counts_and_scores():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.3])
    metrics = premium.classification_metrics(y_true, y_pred, score)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 3)
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5
    assert metrics["roc_auc"] == 0.875


def test_classifier_pipeline_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = (X["a"] > 0.8).astype(int).to_numpy()
    pipeline = premium.build_classifier(0, 20).fit(X, y)
    assert pipeline.predict(X).shape == (200,)
