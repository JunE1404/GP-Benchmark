# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .RegressionDataset import RegressionDataset


class UCIWineQuality(RegressionDataset):
    def __init__(self, features=None, targets=None, feature_types=None):
        wine_quality = fetch_ucirepo(id=186)
        data: Any = wine_quality
        features = data.data.features.to_numpy()
        targets = data.data.targets.to_numpy()
        feature_types = ["con"] * features.shape[1]
        super().__init__(features, targets, feature_types)
