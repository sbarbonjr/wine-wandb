"""Checkpoint 4 (first half): evaluate the chosen artifact on the untouched test split, once.

`--task regression` scores a `wine-pipeline` artifact; `--task premium` scores a
`wine-premium` artifact at its stored threshold and at the default 0.5.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import wandb
import yaml
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score

from wine_wandb import data
from wine_wandb.premium import premium, summarize


def regression_metrics(pipeline, X_test, quality_test):
    predictions = pipeline.predict(X_test)
    return {
        "test/r2": r2_score(quality_test, predictions),
        "test/rmse": float(np.sqrt(mean_squared_error(quality_test, predictions))),
        "test/mae": mean_absolute_error(quality_test, predictions),
    }


def premium_metrics(pipeline, threshold, X_test, quality_test):
    y_test = premium(quality_test)
    score = pipeline.predict_proba(X_test)[:, 1]
    metrics = {"test/pr_auc": float(average_precision_score(y_test, score))}
    for label, tau in (("chosen", threshold), ("default", 0.5)):
        for name, value in summarize(y_test, score, tau).items():
            metrics[f"test/{label}/{name}"] = value
    return metrics


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
            threshold = json.loads((folder / "threshold.json").read_text())["threshold"]
            run.log(premium_metrics(pipeline, threshold, X_test, quality_test))
        else:
            run.log(regression_metrics(pipeline, X_test, quality_test))


if __name__ == "__main__":
    main()
