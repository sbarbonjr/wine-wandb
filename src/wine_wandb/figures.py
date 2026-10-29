"""Figures and numbers for the slides: dataset overview and the test-split confusion matrix.

Fits the premium-wine classifier on the training split and reports the predicted classes on
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

from wine_wandb import data
from wine_wandb.premium import PREMIUM_MIN_QUALITY, build_classifier, classification_metrics, premium

BLUE, GOLD = "#12355B", "#C98A2E"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})


def figure_overview(df, importances, path):
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


def figure_confusion(tn, fp, fn, tp, path):
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap="Blues", vmax=matrix.max() * 1.1)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=11,
                color="white" if value > matrix.max() / 2 else BLUE, fontweight="bold")
    ax.set_xticks([0, 1], ["standard", "premium"])
    ax.set_yticks([0, 1], ["standard", "premium"])
    ax.set(xlabel="predicted", ylabel="actual", title="Test split")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--data", default="data/winequality-red.csv")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = data.load(data.download(args.data))
    X_train, X_test, q_train, q_test = data.split(df, cfg["test_size"], cfg["seed"])
    y_train, y_test = premium(q_train), premium(q_test)

    pipeline = build_classifier(cfg["seed"], cfg["n_estimators"]).fit(X_train, y_train)
    metrics = classification_metrics(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1])
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
        "test": metrics,
        "top_features": [[name, float(value)] for name, value in importances],
        "alcohol_median_premium": float(df.loc[premium(df["quality"]) == 1, "alcohol"].median()),
        "alcohol_median_standard": float(df.loc[premium(df["quality"]) == 0, "alcohol"].median()),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))

    figure_overview(df, importances, out / "wine_overview.pdf")
    figure_confusion(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"], out / "wine_confusion_matrix.pdf")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
