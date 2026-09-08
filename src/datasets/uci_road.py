# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import GetLocal, RegressionDataset
import pandas as pd


class UCIRoad(RegressionDataset):
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
            data = pd.read_csv("src/datasets/localfiles/UCIRoad.txt", header=None)
            features = data.iloc[:,[0,2,3]].to_numpy()
            targets = data.iloc[:,[1]].to_numpy()
        else:
            print("local")
            features = f_local
            targets = t_local
        print(features.shape[1])
        feature_exclude = [0]
        feature_types = [
            "con",
            "con"
        ]
        super().__init__(features=features, targets=targets, feature_types=feature_types, excluded_feature_indeces=feature_exclude)
