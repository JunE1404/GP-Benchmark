# src/Datasets/__init__.py
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import GetLocal, RegressionDataset
import pandas as pd


class UCIRoad(RegressionDataset):
    def __init__(self) -> None:
        """Load the UCI Road dataset.

        Uses the local cache when present, otherwise reads
        ``src/datasets/localfiles/UCIRoad.txt``. Column 1 is the target; columns
        0, 2 and 3 are selected as features and column 0 is then dropped, so
        the two retained features are continuous.
        """
        f_local, t_local = GetLocal(self)
        if f_local is None or t_local is None:
            data = pd.read_csv("src/datasets/localfiles/UCIRoad.txt", header=None)
            features = data.iloc[:,[0,2,3]].to_numpy()
            targets = data.iloc[:,[1]].to_numpy()
        else:
            features = f_local
            targets = t_local
        feature_exclude = [0]
        feature_types = [
            "con",
            "con"
        ]
        super().__init__(features=features, targets=targets, feature_types=feature_types, excluded_feature_indeces=feature_exclude)
