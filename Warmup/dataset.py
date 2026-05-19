from typing import Any

import numpy as np
import torch
from numpy._typing import NDArray
from ucimlrepo import fetch_ucirepo


class Data:
    def __init__(
        self, features: NDArray | torch.Tensor, targets: NDArray | torch.Tensor
    ):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        if targets.dim() == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(-1)

        # Move to CUDA if available
        if torch.cuda.is_available():
            features = features.cuda()
            targets = targets.cuda()

        self.features = features
        self.targets = targets

    def normalize(self):
        X = self.features[:]
        X = X - X.min(0)[0]
        X = 2 * (X / X.max(0)[0]) - 1
        y = self.targets
        self.features = X
        self.targets = y


class Dataset:
    def __init__(self):
        self.data: Data = Data(np.array([]), np.array([]))

    def load_from_uci(self, dataset_name: str | None, dataset_id: int | None):

        if dataset_name is not None and dataset_id is None:
            ds = fetch_ucirepo(name=dataset_name)
        elif dataset_name is None and dataset_id is not None:
            ds = fetch_ucirepo(id=dataset_id)
        else:
            raise Exception(
                "load_from_uci: You must specify a dataset name OR dataset id."
            )

        data: Any = ds

        features = data.data.features
        targets = data.data.targets

        if hasattr(features, "to_numpy"):
            features = features.to_numpy()
        if hasattr(targets, "to_numpy"):
            targets = targets.to_numpy()

        self.data = Data(features, targets)

    def set_data(self, features: NDArray, targets: NDArray):
        if len(features) == 0 or len(targets) == 0:
            raise Exception("Features and targets must not be empty.")

        self.data = Data(features, targets)

    def get_split(
        self, split: float, shuffle_data: bool = True, seed: int | None = None
    ):
        if len(self.data.features) != 0 and len(self.data.targets) != 0:
            rng = np.random.RandomState(seed)
            features = self.data.features
            targets = self.data.targets

            n = features.shape[0]
            indices = np.arange(n)
            if shuffle_data:
                indices = rng.permutation(n)

            split_idx = int(n * split)

            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

            return Data(features[train_idx], targets[train_idx]), Data(
                features[test_idx], targets[test_idx]
            )
        return Data(np.array([]), np.array([])), Data(np.array([]), np.array([]))
