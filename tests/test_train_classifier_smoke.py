import json

import numpy as np
import pandas as pd

from wine_wandb import data, premium, train_classifier


def _synthetic_csv(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    df[data.TARGET] = np.round(5 + 2 * df["a"]).clip(3, 8)
    csv = tmp_path / "wine.csv"
    df.to_csv(csv, sep=";", index=False)
    return csv


def test_classifier_run_completes_with_wandb_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    csv = _synthetic_csv(tmp_path)
    config = tmp_path / "classifier.yaml"
    config.write_text("project: test\nseed: 1\ntest_size: 0.25\ncv_folds: 3\nn_estimators: 20\n")
    train_classifier.run(str(config), str(csv))


def test_out_of_fold_scores_are_probabilities():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = (X["a"] > 1).astype(int).to_numpy()
    scores = premium.out_of_fold_scores(premium.build_classifier(0, 20), X, y, folds=3, seed=0)
    assert scores.shape == (200,)
    assert scores.min() >= 0 and scores.max() <= 1


def test_threshold_file_round_trip(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 0.24}))
    assert json.loads(path.read_text())["threshold"] == 0.24
