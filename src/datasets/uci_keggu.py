# src/Datasets/__init__.py
from jax.numpy import float64
from typing import Any

from ucimlrepo import fetch_ucirepo

from .regression_dataset import GetLocal, RegressionDataset
import pandas as pd


class UCIKeggu(RegressionDataset):
    def __init__(self) -> None:
        """Load the KEGG Undersampling dataset used for the benchmark.

        Uses the local cache when present, otherwise reads
        ``src/datasets/localfiles/KEGGu.data``, taking columns 1-24 as
        continuous features and column 26 as the target.
        """
        f_local, t_local = GetLocal(self)
        if f_local is None or t_local is None:
            data = pd.read_csv("src/datasets/localfiles/KEGGu.data", header=None, na_values="?")
            f_indeces = [x for x in range(1,25)]
            features = data.iloc[:,f_indeces].to_numpy(dtype=float64, na_value=0)
            targets = data.iloc[:,[26]].to_numpy(dtype=float64,na_value=0)
        else:
            print("local")
            features = f_local
            targets = t_local
        feature_types = [
            "con" for i in range(24)
        ]
        super().__init__(features=features, targets=targets, feature_types=feature_types)
