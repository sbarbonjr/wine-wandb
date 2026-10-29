"""Data access: download, checksum, load, and a reproducible train/test split."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET = "quality"


def download(path: str | Path, url: str = URL) -> Path:
    """Download the CSV to `path` unless it already exists."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def sha256(path: str | Path) -> str:
    """Checksum logged with every run to identify the exact dataset version."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str | Path) -> pd.DataFrame:
    """The UCI file is semicolon-separated."""
    return pd.read_csv(path, sep=";")


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Fixed split: the test partition is only used by `evaluate.py`."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)
