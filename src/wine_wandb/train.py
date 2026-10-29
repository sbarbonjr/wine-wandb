"""Checkpoints 1-3: baseline run, cross-validated score, versioned pipeline artifact.

Run once with the baseline configuration, or call `run()` from a sweep agent;
sweep parameters override the values read from the YAML file.
"""
import argparse
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import wandb
import yaml
from sklearn.model_selection import KFold, cross_val_score

from wine_wandb import data
from wine_wandb.pipeline import HYPERPARAMETERS, build_pipeline

ARTIFACT_NAME = "wine-pipeline"


def run(config_path: str = "configs/baseline.yaml", data_path: str = "data/winequality-red.csv"):
    defaults = yaml.safe_load(Path(config_path).read_text())
    data.download(data_path)

    with wandb.init(project=defaults["project"], config=defaults) as active:
        config = active.config
        active.config.update({"data_sha256": data.sha256(data_path)}, allow_val_change=True)

        df = data.load(data_path)
        X_train, _, y_train, _ = data.split(df, config["test_size"], config["seed"])

        hyperparameters = {key: config[key] for key in HYPERPARAMETERS}
        pipeline = build_pipeline(config["seed"], **hyperparameters)

        folds = KFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
        start = time.perf_counter()
        scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="r2")
        active.log(
            {
                "cv/r2_mean": float(np.mean(scores)),
                "cv/r2_std": float(np.std(scores)),
                "cv/seconds": time.perf_counter() - start,
            }
        )

        pipeline.fit(X_train, y_train)
        artifact = wandb.Artifact(
            ARTIFACT_NAME,
            type="model",
            metadata={**hyperparameters, "cv_r2_mean": float(np.mean(scores))},
        )
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(pipeline, Path(tmp) / "pipeline.joblib")
            artifact.add_file(str(Path(tmp) / "pipeline.joblib"))
            active.log_artifact(artifact)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    args = parser.parse_args(argv)
    run(args.config, args.data)


if __name__ == "__main__":
    main()
