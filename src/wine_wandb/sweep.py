"""Checkpoint 2: compare configurations with a W&B sweep (grid search by default)."""
import argparse
from pathlib import Path

import wandb
import yaml

from wine_wandb import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="configs/sweep.yaml")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    parser.add_argument("--count", type=int, default=None, help="maximum number of runs")
    args = parser.parse_args(argv)

    project = yaml.safe_load(Path(args.config).read_text())["project"]
    sweep_id = wandb.sweep(yaml.safe_load(Path(args.sweep).read_text()), project=project)
    wandb.agent(
        sweep_id,
        function=lambda: train.run(args.config, args.data),
        project=project,
        count=args.count,
    )


if __name__ == "__main__":
    main()
