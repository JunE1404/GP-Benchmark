# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import GetLocal, RegressionDataset


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
        f_local, t_local = GetLocal(self)
        if f_local is None or t_local is None:
            print("dl")
            park = fetch_ucirepo(id=189)
            data: Any = park
            features = data.data.features.to_numpy()
            targets = data.data.targets.to_numpy()
        else:
            print("local")
            features = f_local
            targets = t_local

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
