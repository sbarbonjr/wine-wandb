"""Checkpoint 4 (first half): evaluate the chosen artifact on the untouched test split, once."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import wandb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_wandb import data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="wine-pipeline:latest")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    with wandb.init(project=cfg["project"], job_type="evaluate", config=cfg) as run:
        folder = run.use_artifact(args.artifact).download()
        pipeline = joblib.load(Path(folder) / "pipeline.joblib")

        df = data.load(data.download(args.data))
        _, X_test, _, y_test = data.split(df, cfg["test_size"], cfg["seed"])
        predictions = pipeline.predict(X_test)
        run.log(
            {
                "test/r2": r2_score(y_test, predictions),
                "test/rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "test/mae": mean_absolute_error(y_test, predictions),
            }
        )


if __name__ == "__main__":
    main()
