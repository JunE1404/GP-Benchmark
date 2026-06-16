# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .RegressionDataset import RegressionDataset


class UCIParkinsonsTelemonitoring(RegressionDataset):
    def __init__(self, features=None, targets=None, feature_types=None):
        park = fetch_ucirepo(id=189)
        data: Any = park
        features = data.data.features.to_numpy()
        targets = data.data.targets.to_numpy()
        feature_types = [
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "con",
            "cat",
        ]
        super().__init__(features, targets, feature_types)
