"""The model: preprocessing and predictor as a single scikit-learn estimator."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HYPERPARAMETERS = ("n_estimators", "max_depth", "learning_rate", "min_samples_split")


def build_pipeline(seed: int = 42, **hyperparameters) -> Pipeline:
    """StandardScaler followed by a gradient boosting regressor."""
    unknown = set(hyperparameters) - set(HYPERPARAMETERS)
    if unknown:
        raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=seed, **hyperparameters)),
        ]
    )
