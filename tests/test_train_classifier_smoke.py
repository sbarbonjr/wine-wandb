import numpy as np
import pandas as pd

from wine_wandb import data, train_classifier


def test_classifier_run_completes_with_wandb_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    df[data.TARGET] = np.round(5 + 2 * df["a"]).clip(3, 8)
    csv = tmp_path / "wine.csv"
    df.to_csv(csv, sep=";", index=False)

    config = tmp_path / "classifier.yaml"
    config.write_text("project: test\nseed: 1\ntest_size: 0.25\ncv_folds: 3\nn_estimators: 20\n")
    train_classifier.run(str(config), str(csv))
