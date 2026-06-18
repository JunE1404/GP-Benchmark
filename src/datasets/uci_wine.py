# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import RegressionDataset


class UCIWineQuality(RegressionDataset):
    def __init__(self):
        """Load the UCI Wine Quality dataset (id=186).

        Fetches the dataset from the UCI repository. All features are
        treated as continuous.

        Args:
            features: Ignored; data is fetched from UCI.
            targets: Ignored; data is fetched from UCI.
            feature_types: Ignored; all features are treated as continuous.
        """
        wine_quality = fetch_ucirepo(id=186)
        data: Any = wine_quality
        features = data.data.features.to_numpy()
        targets = data.data.targets.to_numpy()
        feature_types = ["con"] * features.shape[1]
        super().__init__(features, targets, feature_types)
