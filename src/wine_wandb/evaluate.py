"""Checkpoint 4 (first half): evaluate the chosen artifact on the untouched test split, once.

`--task regression` scores a `wine-pipeline` artifact; `--task premium` scores a
`wine-premium` artifact using its predicted classes.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import wandb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_wandb import data
from wine_wandb.premium import classification_metrics, premium


def regression_metrics(pipeline, X_test, quality_test):
    predictions = pipeline.predict(X_test)
    return {
        "test/r2": r2_score(quality_test, predictions),
        "test/rmse": float(np.sqrt(mean_squared_error(quality_test, predictions))),
        "test/mae": mean_absolute_error(quality_test, predictions),
    }


def premium_metrics(pipeline, X_test, quality_test):
    y_test = premium(quality_test)
    metrics = classification_metrics(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1])
    return {f"test/{name}": value for name, value in metrics.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["regression", "premium"], default="regression")
    parser.add_argument("--artifact", default=None, help="default: wine-pipeline:latest or wine-premium:latest")
    parser.add_argument("--config", default=None, help="default: configs/baseline.yaml or classifier.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    args = parser.parse_args(argv)

    artifact_name = args.artifact or ("wine-premium:latest" if args.task == "premium" else "wine-pipeline:latest")
    config_path = args.config or ("configs/classifier.yaml" if args.task == "premium" else "configs/baseline.yaml")
    cfg = yaml.safe_load(Path(config_path).read_text())

    with wandb.init(project=cfg["project"], job_type=f"evaluate-{args.task}", config=cfg) as run:
        folder = Path(run.use_artifact(artifact_name).download())
        pipeline = joblib.load(folder / "pipeline.joblib")

        df = data.load(data.download(args.data))
        _, X_test, _, quality_test = data.split(df, cfg["test_size"], cfg["seed"])
        if args.task == "premium":
            run.log(premium_metrics(pipeline, X_test, quality_test))
        else:
            run.log(regression_metrics(pipeline, X_test, quality_test))


if __name__ == "__main__":
    main()
