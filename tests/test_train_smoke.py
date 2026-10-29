import numpy as np
import pandas as pd

from wine_wandb import data, train


def test_train_runs_without_network_or_account(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    df[data.TARGET] = 5 + df["a"]
    csv = tmp_path / "wine.csv"
    df.to_csv(csv, sep=";", index=False)

    config = tmp_path / "config.yaml"
    config.write_text(
        "project: test\nseed: 1\ntest_size: 0.25\ncv_folds: 3\n"
        "n_estimators: 10\nmax_depth: 2\nlearning_rate: 0.1\nmin_samples_split: 2\n"
    )
    train.run(str(config), str(csv))
