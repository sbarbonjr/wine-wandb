import numpy as np

from wine_wandb import decision


def test_premium_label_threshold():
    assert decision.premium([5, 6, 7, 8]).tolist() == [0, 0, 1, 1]


def test_cost_weights_false_negatives_more():
    assert decision.cost(fp=2, fn=1) == 2 * decision.COST_FALSE_POSITIVE + decision.COST_FALSE_NEGATIVE


def test_cost_optimal_threshold_is_below_default():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(int)
    score = np.clip(0.35 * y + rng.normal(0.25, 0.15, 2000), 0, 1)
    tau, _ = decision.choose_threshold(y, score, np.round(np.arange(0.02, 0.99, 0.01), 2))
    assert tau < 0.5
