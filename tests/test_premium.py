import numpy as np

from wine_wandb import premium


def test_premium_label_threshold():
    assert premium.premium([5, 6, 7, 8]).tolist() == [0, 0, 1, 1]


def test_cost_weights_false_negatives_more():
    assert premium.cost(fp=2, fn=1) == 2 * premium.COST_FALSE_POSITIVE + premium.COST_FALSE_NEGATIVE


def test_cost_optimal_threshold_is_below_default():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(int)
    score = np.clip(0.35 * y + rng.normal(0.25, 0.15, 2000), 0, 1)
    tau, _ = premium.choose_threshold(y, score, np.round(np.arange(0.02, 0.99, 0.01), 2))
    assert tau < 0.5
