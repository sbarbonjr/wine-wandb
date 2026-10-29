"""Task 2: a tracked run for the premium-wine classifier (quality >= 7).

Mirrors `train.py`: out-of-fold scores on the training split, a decision threshold chosen
from the costs, and one artifact that bundles the pipeline with that threshold.
"""
import argparse
import json
import tempfile
from pathlib import Path

import joblib
import wandb
import yaml
from sklearn.metrics import average_precision_score

from wine_wandb import data
from wine_wandb.premium import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    THRESHOLD_GRID,
    build_classifier,
    choose_threshold,
    cost,
    counts,
    out_of_fold_scores,
    premium,
)

ARTIFACT_NAME = "wine-premium"


def run(config_path: str = "configs/classifier.yaml", data_path: str = "data/winequality-red.csv"):
    defaults = yaml.safe_load(Path(config_path).read_text())
    data.download(data_path)

    with wandb.init(project=defaults["project"], job_type="train-classifier", config=defaults) as active:
        config = active.config
        active.config.update(
            {
                "data_sha256": data.sha256(data_path),
                "cost_false_positive": COST_FALSE_POSITIVE,
                "cost_false_negative": COST_FALSE_NEGATIVE,
            },
            allow_val_change=True,
        )

        df = data.load(data_path)
        X_train, _, quality_train, _ = data.split(df, config["test_size"], config["seed"])
        y_train = premium(quality_train)

        pipeline = build_classifier(config["seed"], config["n_estimators"])
        scores = out_of_fold_scores(pipeline, X_train, y_train, config["cv_folds"], config["seed"])
        threshold, _ = choose_threshold(y_train, scores, THRESHOLD_GRID)
        _, fp, fn, _ = counts(y_train, scores, threshold)
        active.log(
            {
                "cv/pr_auc": float(average_precision_score(y_train, scores)),
                "cv/threshold": threshold,
                "cv/cost": cost(fp, fn),
            }
        )

        pipeline.fit(X_train, y_train)
        artifact = wandb.Artifact(
            ARTIFACT_NAME,
            type="model",
            metadata={
                "threshold": threshold,
                "cost_false_positive": COST_FALSE_POSITIVE,
                "cost_false_negative": COST_FALSE_NEGATIVE,
                "n_estimators": config["n_estimators"],
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(pipeline, Path(tmp) / "pipeline.joblib")
            (Path(tmp) / "threshold.json").write_text(json.dumps({"threshold": threshold}))
            artifact.add_file(str(Path(tmp) / "pipeline.joblib"))
            artifact.add_file(str(Path(tmp) / "threshold.json"))
            active.log_artifact(artifact)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    args = parser.parse_args(argv)
    run(args.config, args.data)


if __name__ == "__main__":
    main()
