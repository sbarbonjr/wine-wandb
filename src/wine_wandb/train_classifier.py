"""Task 2: a tracked run for the premium-wine classifier (quality >= 7).

Mirrors `train.py`: cross-validated scores on the training split, then a final fit that is
logged as a versioned pipeline artifact. The predicted class is the decision.
"""
import argparse
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import wandb
import yaml
from sklearn.model_selection import StratifiedKFold, cross_validate

from wine_wandb import data
from wine_wandb.premium import build_classifier, premium

ARTIFACT_NAME = "wine-premium"
SCORING = {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}


def run(config_path: str = "configs/classifier.yaml", data_path: str = "data/winequality-red.csv"):
    defaults = yaml.safe_load(Path(config_path).read_text())
    data.download(data_path)

    with wandb.init(project=defaults["project"], job_type="train-classifier", config=defaults) as active:
        config = active.config
        active.config.update({"data_sha256": data.sha256(data_path)}, allow_val_change=True)

        df = data.load(data_path)
        X_train, _, quality_train, _ = data.split(df, config["test_size"], config["seed"])
        y_train = premium(quality_train)

        pipeline = build_classifier(config["seed"], config["n_estimators"])
        folds = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
        start = time.perf_counter()
        result = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=SCORING)
        active.log(
            {
                "cv/f1_mean": float(np.mean(result["test_f1"])),
                "cv/f1_std": float(np.std(result["test_f1"])),
                "cv/precision_mean": float(np.mean(result["test_precision"])),
                "cv/recall_mean": float(np.mean(result["test_recall"])),
                "cv/roc_auc_mean": float(np.mean(result["test_roc_auc"])),
                "cv/seconds": time.perf_counter() - start,
            }
        )

        pipeline.fit(X_train, y_train)
        artifact = wandb.Artifact(
            ARTIFACT_NAME,
            type="model",
            metadata={"n_estimators": config["n_estimators"], "cv_f1_mean": float(np.mean(result["test_f1"]))},
        )
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(pipeline, Path(tmp) / "pipeline.joblib")
            artifact.add_file(str(Path(tmp) / "pipeline.joblib"))
            active.log_artifact(artifact)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    args = parser.parse_args(argv)
    run(args.config, args.data)


if __name__ == "__main__":
    main()
