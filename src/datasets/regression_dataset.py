from abc import ABC
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from sympy.logic.boolalg import Tuple
from torch import Tensor, mean, std, tensor

FeatureTypes = Literal["con", "cat"]


class RegressionDataset:
    def __init__(
        self,
        features: NDArray | Tensor = None,
        targets: NDArray | Tensor = None,
        feature_types: list[FeatureTypes] = None,
    ) -> None:
        """Initialize the dataset with features, targets, and their types.

        Args:
            features: Input features, either as a NumPy array or PyTorch tensor.
            targets: Target values, either as a NumPy array or PyTorch tensor.
            feature_types: List of feature type labels ("con" for continuous, "cat" for categorical).
        """
        self.features = self._convert_to_tensor(features)
        self.targets = self._convert_to_tensor(targets).flatten()
        self.feature_types = feature_types

        if self.features.ndim == 1:
            self.features = self.features.unsqueeze(-1)

    def __str__(self) -> str:
        """Return the class name as the string representation."""
        return f"{self.__class__.__name__}"

    @staticmethod
    def _convert_to_tensor(a: NDArray | Tensor) -> Tensor:
        """Convert a NumPy array or PyTorch tensor to a float32 tensor.

        Args:
            a: Input array or tensor.

        Returns:
            A float32 PyTorch tensor.
        """
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

    def _standardize_data(
        self,
        i_features: Tensor,
        i_targets: Tensor,
        f_means: Tensor,
        f_stds: Tensor,
        t_mean: float,
        t_std: float,
        standardize_parts: tuple[bool, bool],
    ) -> tuple[Tensor, Tensor]:
        """Standardize features and/or targets using provided statistics.

        Args:
            i_features: Input features to standardize.
            i_targets: Input targets to standardize.
            f_means: Feature means for standardization.
            f_stds: Feature standard deviations for standardization.
            t_mean: Target mean for standardization.
            t_std: Target standard deviation for standardization.
            standardize_parts: Tuple of (standardize_features, standardize_targets) flags.

        Returns:
            Tuple of (standardized_features, standardized_targets).
        """
        return_features = i_features
        return_targets = i_targets
        if standardize_parts[0]:
            return_features = (i_features - f_means) / f_stds
        if standardize_parts[1]:
            return_targets = (i_targets - t_mean) / t_std

        return return_features, return_targets

    def get_onehot_encoded_features(self):
        """Return features with categorical columns one-hot encoded.

        Continuous features are kept as-is; categorical features are converted
        to one-hot vectors and concatenated alongside the continuous features.

        Returns:
            Tensor of shape (n_samples, n_encoded_features) with one-hot encoded features.
        """
        return_features = self.features

        if return_features.shape[0] == 0:
            return torch.empty(0, 0)

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

        return_features = torch.cat([cont_features, cat_encoded], dim=1)

        return return_features

    def cuda(self):
        """Move features and targets to GPU if CUDA is available.

        Returns:
            Self, to allow method chaining.
        """
        if torch.cuda.is_available():
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()

            return self

    def cpu(self):
        """Move features and targets to CPU.

        Returns:
            Self, to allow method chaining.
        """
        self.features = self.features.cpu()
        self.targets = self.targets.cpu()

        return self

    def _get_dataset_statistics(
        self,
        i_features: Tensor,
        i_targets: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Compute mean and standard deviation of features and targets.

        Continuous features are used for the feature statistics (one-hot
        columns are excluded).

        Args:
            i_features: Input features to compute statistics for.
            i_targets: Input targets to compute statistics for.

        Returns:
            Tuple of ((feature_means, feature_stds), (target_mean, target_std)).
        """
        train_X_means = torch.zeros(self.features.shape[1])
        train_X_stds = torch.ones(self.features.shape[1])

        n_one_hot = (
            self.features.shape[1] - (np.array(self.feature_types) == "cat").sum()
        )
        features_continuous = self.features[:, :-n_one_hot]
        train_X_means[:-n_one_hot] = mean(features_continuous, dim=0)
        train_X_stds[:-n_one_hot] = std(features_continuous, dim=0)

        # Fix stand of one hot features & re-check cont & cat order

        x_means = mean(i_features, dim=0)
        x_stds = std(i_features, dim=0)
        y_mean = mean(i_targets, dim=0)
        y_std = std(i_targets, dim=0)

        return (x_means, x_stds), (y_mean, y_std)

    def get_data_split(
        self,
        split_fractions: tuple[float, float, float],
        standardize_data_splits: tuple[
            tuple[bool, bool], tuple[bool, bool], tuple[bool, bool]
        ],
        shuffle_data: bool,
        shuffle_seed: float | None,
    ) -> tuple[
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        tuple[Tensor, Tensor],
    ]:
        """Split the dataset into train, validation, and test sets with optional standardization.

        The data is first one-hot encoded, then split according to the provided
        fractions, and each split is optionally standardized using statistics
        computed from the training split.

        Args:
            split_fractions: Tuple of (train, val, test) fractions that sum to 1.
            standardize_data_splits: For each split, a tuple of (standardize_features, standardize_targets).
            shuffle_data: Whether to randomly shuffle the data before splitting.
            shuffle_seed: Random seed for shuffling; if None, a random seed is used.

        Returns:
            Tuple of ((train_features, train_targets), (val_features, val_targets),
                      (test_features, test_targets)).

        Raises:
            ValueError: If features or targets are empty.
        """
        if len(self.features) == 0 or len(self.targets) == 0:
            raise ValueError(
                f"Cannot split dataset '{self.__class__.__name__}': features or targets are empty"
            )
        rng = np.random.RandomState(shuffle_seed)
        features = self.get_onehot_encoded_features()
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

        train_features = features[train_idx]
        train_targets = targets[train_idx]

        val_features = features[val_idx]
        val_targets = targets[val_idx]

        test_features = features[test_idx]
        test_targets = targets[test_idx]

        (t_x_means, t_x_stds), (t_y_mean, t_y_std) = self._get_dataset_statistics(
            train_features, train_targets
        )

        st_train = self._standardize_data(
            train_features,
            train_targets,
            f_means=t_x_means,
            f_stds=t_x_stds,
            t_mean=t_y_mean,
            t_std=t_y_std,
            standardize_parts=standardize_data_splits[0],
        )
        st_val = self._standardize_data(
            val_features,
            val_targets,
            f_means=t_x_means,
            f_stds=t_x_stds,
            t_mean=t_y_mean,
            t_std=t_y_std,
            standardize_parts=standardize_data_splits[1],
        )
        st_test = self._standardize_data(
            test_features,
            test_targets,
            f_means=t_x_means,
            f_stds=t_x_stds,
            t_mean=t_y_mean,
            t_std=t_y_std,
            standardize_parts=standardize_data_splits[2],
        )

        return (st_train, st_val, st_test), (t_y_mean, t_y_std)
