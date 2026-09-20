# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import GetLocal, RegressionDataset


class UCIParkinsonsTelemonitoring(RegressionDataset):
    def __init__(self) -> None:
        """Load the UCI Parkinsons Telemonitoring dataset (id=189).

        Uses the local cache when present, otherwise fetches the dataset from
        the UCI repository. The first 18 features are continuous and the last
        is categorical.
        """
        f_local, t_local = GetLocal(self)
        if f_local is None or t_local is None:
            park = fetch_ucirepo(id=189)
            data: Any = park
            features = data.data.features.to_numpy()
            targets = data.data.targets.to_numpy()
        else:
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
