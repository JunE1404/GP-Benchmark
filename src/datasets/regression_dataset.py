from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, mean, std, tensor

FeatureTypes = Literal["con", "cat"]


class Dataset:
    features: Tensor
    targets: Tensor
    feature_types: list[FeatureTypes]

    def __init__(
        self,
        features: NDArray | Tensor,
        targets: NDArray | Tensor,
        feature_types: list[FeatureTypes],
    ) -> None:
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        self.features = features
        self.targets = targets.flatten()
        self.feature_types = feature_types

    def standardize_data(self, f: bool, t: bool):
        # statelessness preferred, return seperate data after standardization
        self.apply_onehot_encoding()

        (f_means, f_stds), (t_mean, t_std) = self.get_dataset_statistics()

        if f:
            self.features = (self.features - f_means) / f_stds
        if t:
            self.targets = (self.targets - t_mean) / t_std

    def apply_onehot_encoding(self):
        if self.features.shape[0] == 0:
            return

        cat_mask = np.array(self.feature_types) == "cat"
        cont_mask = ~cat_mask
        cont_features = self.features[:, cont_mask]
        cat_features = self.features[:, cat_mask]

        encoded_cols = []
        for i in range(cat_features.shape[1]):
            col = cat_features[:, i].long()
            n_classes = int(col.max().item() + 1)
            one_hot = torch.nn.functional.one_hot(col, num_classes=n_classes)
            encoded_cols.append(one_hot)

        cat_encoded = (
            torch.cat(encoded_cols, dim=1)
            if encoded_cols
            else torch.empty(self.features.shape[0], 0)
        )

        self.features = torch.cat([cont_features, cat_encoded], dim=1)

    def cuda(self):
        if torch.cuda.is_available():
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()

            return self

    def cpu(self):
        self.features = self.features.cpu()
        self.targets = self.targets.cpu()

        return self

    def get_dataset_statistics(
        self,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:

        # train_X_means = torch.zeros(self.features.shape[1])
        # train_X_stds = torch.ones(self.features.shape[1])

        # n_one_hot = (np.array(self.feature_types) == "cat").sum()
        # features_continuous = self.features[:, n_one_hot:]
        # train_X_means[n_one_hot:] = mean(features_continuous, dim=0)
        # train_X_stds[n_one_hot:] = std(features_continuous, dim=0)
        #
        # #^ Exclude one-hot encoded columns from statistics

        x_means = mean(self.features, dim=0)
        x_stds = std(self.features, dim=0)
        y_mean = mean(self.targets, dim=0)
        y_std = std(self.targets, dim=0)

        return (x_means, x_stds), (y_mean, y_std)


class RegressionDataset(Dataset):
    def __init__(
        self,
        features: NDArray | Tensor,
        targets: NDArray | Tensor,
        feature_types: list[FeatureTypes],
    ) -> None:

        f = self._convert_to_tensor(features)
        t = self._convert_to_tensor(targets).flatten()
        self.feature_types = feature_types

        super().__init__(f, t, feature_types)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    @staticmethod
    def _convert_to_tensor(a: NDArray | Tensor) -> Tensor:
        if isinstance(a, Tensor):
            return a.float() if not a.is_floating_point() else a
        return tensor(a).float()

    @property
    def input_dim(self) -> int:
        """Return the input dimensionality of the dataset."""
        return self.features.shape[1]

    @property
    def output_dim(self) -> int:
        """Return the output dimensionality of the dataset."""
        return self.targets.shape[1] if self.targets.ndim > 1 else 1

    def get_data_split(
        self,
        split_fractions: tuple[float, float, float],
        standardize_data_split_features: tuple[bool, bool, bool],
        standardize_data_split_targets: tuple[bool, bool, bool],
        shuffle_data: bool,
        shuffle_seed: float | None,
    ) -> tuple[Dataset, Dataset, Dataset]:

        if len(self.features) == 0 or len(self.targets) == 0:
            raise ValueError(
                f"Cannot split dataset '{self.__class__.__name__}': features or targets are empty"
            )
        rng = np.random.RandomState(shuffle_seed)
        features = self.features
        targets = self.targets

        n = features.shape[0]
        indices = np.arange(n)
        if shuffle_data:
            indices = rng.permutation(n)

        split_idx_train_val = int(n * split_fractions[0])
        split_idx_val_test = split_idx_train_val + int(n * split_fractions[1])

        train_idx = indices[:split_idx_train_val]
        val_idx = indices[split_idx_train_val:split_idx_val_test]
        test_idx = indices[split_idx_val_test:]

        train = Dataset(
            features=features[train_idx],
            targets=targets[train_idx],
            feature_types=self.feature_types,
        )
        val = Dataset(
            features=features[val_idx],
            targets=targets[val_idx],
            feature_types=self.feature_types,
        )
        test = Dataset(
            features=features[test_idx],
            targets=targets[test_idx],
            feature_types=self.feature_types,
        )

        train.standardize_data(
            standardize_data_split_features[0],
            standardize_data_split_targets[0],
            #
        )
        val.standardize_data(
            standardize_data_split_features[1],
            standardize_data_split_targets[1],
            # user train statistics for standardizing all splits
        )
        test.standardize_data(
            standardize_data_split_features[2], standardize_data_split_targets[2]
        )

        return (train, val, test)
