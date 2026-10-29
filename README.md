# wine-wandb

Red wine quality with [Weights & Biases](https://wandb.ai) experiment tracking.
A small but complete project used in the course *Data-driven Systems Engineering* (440MI / 305SM):
a reproducible baseline, a hyperparameter sweep, versioned model artifacts, and a Registry
promotion.

## One dataset, two tasks

The UCI *Wine Quality* file (red wine, 1,599 samples, 11 physicochemical features) supports two
framings of the same problem. Both are tracked, versioned, and registered the same way.

| | Task 1: regression | Task 2: premium classification |
|---|---|---|
| Question | What is the quality score? | Is the wine premium (quality >= 7)? |
| Target | `quality` (3 to 8) | `quality >= 7` (217 wines, 13.6%) |
| Model | `GradientBoostingRegressor` | `RandomForestClassifier` (300 trees) |
| Selection metric | cross-validated R2 (`KFold`) | out-of-fold cost at the chosen threshold (`StratifiedKFold`) |
| Decision | none | probability and a cost-based threshold |
| Tracked run | `train.py`, `sweep.py` | `train_classifier.py` |
| Artifact | `wine-pipeline` (`pipeline.joblib`) | `wine-premium` (`pipeline.joblib` + `threshold.json`) |
| Used in class for | tracking, sweeps, registry | evaluation, thresholds, decision support |

Both tasks share the data module, the fixed 75/25 split, and the seed. The premium artifact stores
the decision threshold next to the pipeline, so the threshold is a versioned release parameter.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
wandb login            # once; not needed for the tests
make data              # downloads the UCI CSV into data/
make train             # task 1: baseline run
make sweep             # task 1: 81-configuration grid search
make train-classifier  # task 2: premium classifier run
make evaluate ARTIFACT=wine-pipeline:v0           # task 1, test split, once
make evaluate-premium ARTIFACT=wine-premium:v0    # task 2, test split, once
make decision          # threshold analysis; writes reports/ (JSON and PDF figures)
make test              # no network or account required
```

## Layout

| Path | Purpose |
|---|---|
| `configs/baseline.yaml` | Task 1: seed, split, CV folds, default hyperparameters |
| `configs/sweep.yaml` | Task 1: W&B sweep, 3 x 3 x 3 x 3 grid, metric `cv/r2_mean` |
| `configs/classifier.yaml` | Task 2: seed, split, CV folds, number of trees |
| `src/wine_wandb/data.py` | Download, checksum, load, fixed train/test split |
| `src/wine_wandb/pipeline.py` | Task 1: `StandardScaler` + `GradientBoostingRegressor` as one `Pipeline` |
| `src/wine_wandb/premium.py` | Task 2: label, classifier pipeline, costs, out-of-fold scores, threshold choice |
| `src/wine_wandb/train.py` | Task 1: one tracked run (CV score, timing, pipeline artifact) |
| `src/wine_wandb/train_classifier.py` | Task 2: one tracked run (out-of-fold PR-AUC, threshold, pipeline + threshold artifact) |
| `src/wine_wandb/sweep.py` | Creates a sweep and runs an agent that calls `train.run` |
| `src/wine_wandb/evaluate.py` | Scores an artifact on the untouched test split (`--task regression` or `premium`) |
| `src/wine_wandb/register.py` | Links an approved artifact version to a Registry collection |
| `src/wine_wandb/decision.py` | Premium analysis for the slides: test metrics at the default and chosen thresholds, and the figures |
| `tests/` | Unit tests and smoke tests (`WANDB_MODE=disabled`) |

## The four checkpoints

1. **Baseline:** `make train` (or `make train-classifier`) with the fixed split and seed.
2. **Compare:** `make sweep`; compare runs in the W&B dashboard (mean and standard deviation of the CV R2).
3. **Version:** every run logs the full pipeline (preprocessing + model) as an artifact, together with the dataset SHA-256 and, automatically, the Git commit.
4. **Approve:** choose a version using cross-validation only, run the matching `evaluate` target once on the test split, then link it to the Registry, for example
   `python -m wine_wandb.register --artifact wine-premium:v0 --collection wandb-registry-model/wine-premium`.
   Rolling back means moving the alias to the previous version.

The test split is read only by `evaluate.py` and `decision.py`. Model and threshold selection use cross-validation on the training split.

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
