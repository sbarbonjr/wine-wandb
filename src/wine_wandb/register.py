"""Checkpoint 4 (second half): link an approved artifact version to a Registry collection."""
import argparse
from pathlib import Path

import wandb
import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="e.g. wine-pipeline:v3")
    parser.add_argument(
        "--collection",
        default="wandb-registry-model/wine-quality",
        help="Registry path: wandb-registry-<registry>/<collection>",
    )
    parser.add_argument("--alias", default="approved")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args(argv)

    project = yaml.safe_load(Path(args.config).read_text())["project"]
    with wandb.init(project=project, job_type="register") as run:
        artifact = run.use_artifact(args.artifact)
        run.link_artifact(artifact, target_path=args.collection, aliases=[args.alias])


if __name__ == "__main__":
    main()
