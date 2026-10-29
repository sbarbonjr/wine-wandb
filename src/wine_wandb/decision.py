"""From a score to a decision: a "premium wine" classifier, a cost-based threshold, and figures.

The regression target `quality` is turned into a binary label (quality >= 7). The decision
threshold is chosen on out-of-fold predictions of the training split and then applied once to
the test split. Everything is deterministic given the seed.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wine_wandb import data

PREMIUM_MIN_QUALITY = 7
COST_FALSE_POSITIVE = 1.0  # a standard wine sold as premium
COST_FALSE_NEGATIVE = 5.0  # a premium wine sold as standard
BLUE, GOLD, GRAY = "#12355B", "#C98A2E", "#4B5563"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})


def premium(quality):
    return (np.asarray(quality) >= PREMIUM_MIN_QUALITY).astype(int)


def build_classifier(seed: int, n_estimators: int = 300) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ]
    )


def counts(y_true, score, tau):
    tn, fp, fn, tp = confusion_matrix(y_true, (score >= tau).astype(int), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def cost(fp, fn, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    return c_fp * fp + c_fn * fn


def summarize(y_true, score, tau):
    tn, fp, fn, tp = counts(y_true, score, tau)
    return {
        "tau": float(tau),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / len(y_true),
        "cost": cost(fp, fn),
    }


def choose_threshold(y_true, score, grid):
    """Threshold minimizing the cost; ties resolved by the median of the minimizers."""
    costs = np.array([cost(*counts(y_true, score, t)[1:3]) for t in grid])
    return float(np.median(grid[costs == costs.min()])), costs


def figure_overview(df, y_train_binary, importances, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    quality = df["quality"].value_counts().sort_index()
    axes[0].bar(quality.index, quality.values,
                color=[GOLD if q >= PREMIUM_MIN_QUALITY else BLUE for q in quality.index])
    axes[0].set(title="Quality scores", xlabel="quality", ylabel="wines")
    names, values = zip(*importances)
    axes[1].barh(names[::-1], values[::-1], color=BLUE)
    axes[1].set(title="Random forest importance", xlabel="impurity-based")
    for flag, label, color in ((0, "standard", BLUE), (1, "premium", GOLD)):
        axes[2].hist(df.loc[premium(df["quality"]) == flag, "alcohol"], bins=18, alpha=0.75,
                     color=color, label=label, density=True)
    axes[2].set(title="Alcohol (% vol.)", xlabel="alcohol", ylabel="density")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_confusion(tn, fp, fn, tp, tau, path):
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap="Blues", vmax=matrix.max() * 1.1)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=11,
                color="white" if value > matrix.max() / 2 else BLUE, fontweight="bold")
    ax.set_xticks([0, 1], ["standard", "premium"])
    ax.set_yticks([0, 1], ["standard", "premium"])
    ax.set(xlabel="predicted", ylabel="actual", title=rf"Test split, $\tau$ = {tau:.2f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_threshold(y_true, score, grid, costs, tau_default, tau_chosen, path):
    precision, recall = [], []
    for t in grid:
        _, fp, fn, tp = counts(y_true, score, t)
        precision.append(tp / (tp + fp) if tp + fp else np.nan)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.4))
    axes[0].plot(grid, precision, color=BLUE, label="precision")
    axes[0].plot(grid, recall, color=GOLD, label="recall")
    axes[0].axvline(tau_default, color=GRAY, ls=":", lw=1)
    axes[0].set(title="Precision and recall versus threshold", xlabel=r"threshold $\tau$")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(grid, costs, color=BLUE)
    axes[1].axvline(tau_default, color=GRAY, ls=":", lw=1)
    default_cost = costs[np.argmin(np.abs(grid - tau_default))]
    chosen_cost = costs[np.argmin(np.abs(grid - tau_chosen))]
    axes[1].plot([tau_default], [default_cost], "o", color=GRAY)
    axes[1].plot([tau_chosen], [chosen_cost], "o", color="#B3261E")
    axes[1].annotate(rf"chosen $\tau$ = {tau_chosen:.2f}", (tau_chosen, chosen_cost),
                     textcoords="offset points", xytext=(-4, 34), fontsize=7, color="#B3261E",
                     arrowprops={"arrowstyle": "-", "color": "#B3261E", "lw": 0.6})
    axes[1].annotate(r"default $\tau$ = 0.5", (tau_default, default_cost),
                     textcoords="offset points", xytext=(12, 22), fontsize=7, color=GRAY,
                     arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.6})
    axes[1].set(title=rf"Cost = FP + {COST_FALSE_NEGATIVE:g} FN (out-of-fold)",
                xlabel=r"threshold $\tau$", ylabel="cost")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = data.load(data.download(args.data))
    X_train, X_test, q_train, q_test = data.split(df, cfg["test_size"], cfg["seed"])
    y_train, y_test = premium(q_train), premium(q_test)

    pipeline = build_classifier(cfg["seed"])
    folds = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    oof = cross_val_predict(pipeline, X_train, y_train, cv=folds, method="predict_proba")[:, 1]

    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    tau_chosen, costs = choose_threshold(y_train, oof, grid)

    pipeline.fit(X_train, y_train)
    score = pipeline.predict_proba(X_test)[:, 1]
    importances = sorted(
        zip(X_train.columns, pipeline.named_steps["model"].feature_importances_),
        key=lambda pair: -pair[1],
    )[:5]

    result = {
        "n": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "premium_total": int(premium(df["quality"]).sum()),
        "premium_rate": float(premium(df["quality"]).mean()),
        "premium_train": int(y_train.sum()),
        "premium_test": int(y_test.sum()),
        "majority_accuracy_test": float(1 - y_test.mean()),
        "pr_auc_test": float(average_precision_score(y_test, score)),
        "c_fp": COST_FALSE_POSITIVE,
        "c_fn": COST_FALSE_NEGATIVE,
        "tau_theory": COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE),
        "tau_chosen_oof": tau_chosen,
        "test_default": summarize(y_test, score, 0.5),
        "test_chosen": summarize(y_test, score, tau_chosen),
        "top_features": [[name, float(value)] for name, value in importances],
        "alcohol_median_premium": float(df.loc[premium(df["quality"]) == 1, "alcohol"].median()),
        "alcohol_median_standard": float(df.loc[premium(df["quality"]) == 0, "alcohol"].median()),
    }
    (out / "decision.json").write_text(json.dumps(result, indent=2))

    figure_overview(df, y_train, importances, out / "wine_overview.pdf")
    default = result["test_default"]
    figure_confusion(default["tn"], default["fp"], default["fn"], default["tp"], 0.5,
                     out / "wine_confusion_matrix.pdf")
    figure_threshold(y_train, oof, grid, costs, 0.5, tau_chosen, out / "wine_threshold_cost.pdf")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
