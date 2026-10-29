import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score

from wine_wandb import data
from wine_wandb.pipeline import build_pipeline


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    df[data.TARGET] = 5 + df["a"] + 0.5 * df["b"] + rng.normal(scale=0.1, size=120)
    return df


def test_split_is_reproducible_and_sized(frame):
    first = data.split(frame, 0.25, 7)
    second = data.split(frame, 0.25, 7)
    assert len(first[1]) == 30
    pd.testing.assert_frame_equal(first[0], second[0])


def test_pipeline_fits_and_predicts(frame):
    X_train, X_test, y_train, _ = data.split(frame)
    pipeline = build_pipeline(n_estimators=20, max_depth=2).fit(X_train, y_train)
    assert pipeline.predict(X_test).shape == (len(X_test),)


def test_pipeline_in_cross_validation(frame):
    X_train, _, y_train, _ = data.split(frame)
    scores = cross_val_score(build_pipeline(n_estimators=20), X_train, y_train, cv=3)
    assert scores.mean() > 0.5


def test_unknown_hyperparameter_is_rejected():
    with pytest.raises(ValueError):
        build_pipeline(not_a_parameter=1)
