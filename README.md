# wine-wandb

Red wine quality regression with [Weights & Biases](https://wandb.ai) experiment tracking.
A small but complete project used in the course *Data-driven Systems Engineering* (440MI / 305SM):
a reproducible baseline, a hyperparameter sweep, a versioned model artifact, and a Registry
promotion.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
wandb login            # once; not needed for the tests
make data              # downloads the UCI CSV into data/
make train             # baseline run (checkpoints 1 and 3)
make sweep             # 81-configuration grid search (checkpoint 2)
make evaluate ARTIFACT=wine-pipeline:v0   # test split, once (checkpoint 4)
make decision          # threshold analysis; writes reports/ (JSON and PDF figures)
make test              # no network or account required
```

## Layout

| Path | Purpose |
|---|---|
| `configs/baseline.yaml` | Seed, split, CV folds, default hyperparameters |
| `configs/sweep.yaml` | W&B sweep: 3 x 3 x 3 x 3 grid, metric `cv/r2_mean` |
| `src/wine_wandb/data.py` | Download, checksum, load, fixed train/test split |
| `src/wine_wandb/pipeline.py` | `StandardScaler` + `GradientBoostingRegressor` as one `Pipeline` |
| `src/wine_wandb/train.py` | One tracked run: CV score, timing, pipeline artifact |
| `src/wine_wandb/sweep.py` | Creates a sweep and runs an agent that calls `train.run` |
| `src/wine_wandb/evaluate.py` | Scores an artifact on the untouched test split |
| `src/wine_wandb/register.py` | Links an approved artifact version to a Registry collection |
| `src/wine_wandb/decision.py` | "Premium wine" classifier (quality >= 7): cost-based threshold chosen on out-of-fold predictions, test metrics, and the figures used in the slides |
| `tests/` | Unit tests and a smoke test (`WANDB_MODE=disabled`) |

## The four checkpoints

1. **Baseline:** `make train` with the fixed split and seed in `baseline.yaml`.
2. **Compare:** `make sweep`; compare runs in the W&B dashboard (mean and standard deviation of the CV R2).
3. **Version:** every run logs the full pipeline (preprocessing + model) as the `wine-pipeline` artifact, together with the dataset SHA-256 and, automatically, the Git commit.
4. **Approve:** choose a version using the CV score only, run `evaluate` once on the test split, then link it to the Registry with `python -m wine_wandb.register --artifact wine-pipeline:vN`. Rolling back means moving the alias to the previous version.

The test split is read only by `evaluate.py`. Model selection uses cross-validation on the training split.

## Using this repository in a course repository

As a Git submodule:

```bash
git submodule add <url-of-this-repository> projects/wine-wandb
git clone --recurse-submodules <url-of-the-course-repository>   # for new clones
git submodule update --init                                     # for existing clones
```

Because W&B records the commit of the repository where the script runs, a run launched inside the submodule points to a commit of this project alone.

## Data

UCI Machine Learning Repository, *Wine Quality* (red, 1,599 samples, 11 physicochemical features, target `quality`).
P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009). Modeling wine preferences by data mining
from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
Check the license on the dataset page before redistributing the file; it is not committed here.
