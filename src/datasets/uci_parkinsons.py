# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import RegressionDataset


class UCIParkinsonsTelemonitoring(RegressionDataset):
    def __init__(self):
        """Load the UCI Parkinsons Telemonitoring dataset (id=189).

        Fetches the dataset from the UCI repository and assigns feature types:
        18 continuous features and 1 categorical feature.

        Args:
            features: Ignored; data is fetched from UCI.
            targets: Ignored; data is fetched from UCI.
            feature_types: Ignored; types are predefined.
        """
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
