from misc.scaffolds import SplitStatistics
from misc.scaffolds import Split
from misc.scaffolds import DatasetData
from misc.scaffolds import StandardisationBools
from typing_extensions import List
from jedi.api import helpers
from abc import ABC
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from sympy.logic.boolalg import Tuple
from torch import Tensor, mean, std, tensor
import misc.helpers as helpers

FeatureTypes = Literal["con", "cat"]


class RegressionDataset:
    def __init__(
        self,
        features: NDArray | Tensor | None = None,
        targets: NDArray | Tensor | None = None,
        feature_types: list[FeatureTypes] | None = None,
        excluded_feature_indeces: list[int] | None = None,
        includes_feature_indeces: list[int] | None = None
    ) -> None:
        """Initialize the dataset with features, targets, and their types.

        Args:
            features: Input features, either as a NumPy array or PyTorch tensor.
            targets: Target values, either as a NumPy array or PyTorch tensor.
            feature_types: Feature type labels, ``"con"`` for continuous and
                ``"cat"`` for categorical, one per retained feature.
            excluded_feature_indeces: Indices to drop. Mutually exclusive with
                ``includes_feature_indeces``.
            includes_feature_indeces: Indices to keep. Mutually exclusive with
                ``excluded_feature_indeces``.

        Raises:
            Exception: If ``targets``, ``features`` or ``feature_types`` are
                missing, if both index lists are given, or if the retained
                feature count does not match ``feature_types``.
        """

        self.features = self._convert_to_tensor(features)
        self.targets = self._convert_to_tensor(targets).flatten()
        self.feature_types = feature_types

        if self.targets == None:
            raise Exception("Targets missing")
        if self.features == None:
            raise Exception("Features missing")
        if self.feature_types == None:
            raise Exception("Feature types missing")

        n_features = self.features.shape[1]
        n_feature_types = len(self.feature_types)
        
        feature_indeces = [x for x in range(n_features)]

        if excluded_feature_indeces != None and includes_feature_indeces != None:
            raise Exception("Inclusion and exclusion lists are mutually exclusive")
        if excluded_feature_indeces != None:
            inlcuded_features_i = [x for x in feature_indeces if x not in excluded_feature_indeces]
        if includes_feature_indeces != None:
            inlcuded_features_i = includes_feature_indeces
        if excluded_feature_indeces== None and includes_feature_indeces==None:
            inlcuded_features_i = feature_indeces

        SaveLocal(self, self.features.float(), self.targets.float())
        self.features = self.features[:,inlcuded_features_i]
        if self.features.shape[1] != n_feature_types:
            raise Exception("Included features to feature type count mismatch")

        if self.features.ndim == 1:
            self.features = self.features.unsqueeze(-1)

    def __str__(self) -> str:
        """Return the class name as the string representation."""
        return f"{self.__class__.__name__}"

    @staticmethod
    def _convert_to_tensor(a: NDArray | Tensor) -> Tensor:
        """Convert a NumPy array or PyTorch tensor to the default float dtype.

        Uses ``torch.get_default_dtype()`` so the whole pipeline can be run in
        float64 once ``torch.set_default_dtype`` has been changed (see the
        ``--float64`` flag in ``main.py``).

        Args:
            a: Input array or tensor.

        Returns:
            A PyTorch tensor in the current default float dtype.
        """
        if isinstance(a, Tensor):
            return a.to(torch.get_default_dtype())
        return tensor(a).to(torch.get_default_dtype())

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
        standardize_parts: StandardisationBools,
    ) -> Split:
        """Standardize features and/or targets using provided statistics.

        Args:
            i_features: Input features to standardize.
            i_targets: Input targets to standardize.
            f_means: Feature means for standardization.
            f_stds: Feature standard deviations for standardization.
            t_mean: Target mean for standardization.
            t_std: Target standard deviation for standardization.
            standardize_parts: Feature/target standardization flags.

        Returns:
            ``Split`` of (standardized_features, standardized_targets).
        """
        return_features = i_features
        return_targets = i_targets
        if standardize_parts.features:
            return_features = (i_features - f_means) / f_stds
        if standardize_parts.targets:
            return_targets = (i_targets - t_mean) / t_std

        return Split(return_features, return_targets)

    def _get_onehot_encoded_features(self) -> Tensor:
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

    def cuda(self) -> RegressionDataset | None:
        """Move features and targets to GPU if CUDA is available.

        Returns:
            ``self`` when CUDA is available, otherwise ``None``.
        """
        if torch.cuda.is_available():
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()

            return self

    def cpu(self) -> RegressionDataset:
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
    ) -> tuple[SplitStatistics, SplitStatistics]:
        """Compute mean and standard deviation of features and targets.

        Continuous features are used for the feature statistics (one-hot
        columns are excluded).

        Args:
            i_features: Input features to compute statistics for.
            i_targets: Input targets to compute statistics for.

        Returns:
            Tuple of ((feature_means, feature_stds), (target_mean, target_std)).
        """
        x_means = mean(i_features, dim=0)
        x_stds = std(i_features, dim=0)
        y_mean = mean(i_targets, dim=0)
        y_std = std(i_targets, dim=0)

        return SplitStatistics(mean=x_means, std=x_stds), SplitStatistics(mean=y_mean, std=y_std)

    def _deduplicate(self, features: Tensor, targets: Tensor, strategy: str) -> tuple[Tensor, Tensor]:
        """Remove duplicate feature rows, optionally aggregating their targets.

        Duplicate inputs make the covariance matrix exactly rank-deficient and,
        under a random split, also leak the same point into train and test.
        Removing them here (before the split) addresses both.

        Args:
            features: Model inputs of shape (n_samples, n_features).
            targets: Targets aligned with ``features``.
            strategy: ``"none"`` (no-op), ``"mean"`` (average the targets of
                duplicate rows) or ``"first"`` (keep the first occurrence).

        Returns:
            ``(features, targets)`` with duplicate rows removed.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        if strategy == "none":
            return features, targets
        if strategy not in ("mean", "first"):
            raise ValueError(f"Unknown deduplication strategy: '{strategy}'")

        unique, inverse, counts = torch.unique(
            features, dim=0, return_inverse=True, return_counts=True
        )
        if strategy == "first":
            first = torch.full(
                (unique.shape[0],), features.shape[0], dtype=torch.long, device=features.device
            )
            first.scatter_reduce_(
                0, inverse, torch.arange(features.shape[0], device=features.device), reduce="amin"
            )
            return unique, targets[first]

        target_sum = torch.zeros(
            unique.shape[0], *targets.shape[1:], dtype=targets.dtype, device=targets.device
        )
        target_sum.index_add_(0, inverse, targets)
        divisor = counts.reshape(-1, *([1] * (targets.dim() - 1))).to(targets.dtype)
        return unique, target_sum / divisor

    def get_data_splits(
        self,
        split_fractions_argument: str,
        standardize_data_splits_argument: str,
        shuffle_data: bool,
        shuffle_seed: float | None,
        deduplicate: str = "none",
    ) -> DatasetData:
        """Split the dataset into train, validation, and test sets with optional standardization.

        The data is first one-hot encoded, then split according to the provided
        fractions, and each split is optionally standardized using statistics
        computed from the training split.

        Args:
            split_fractions_argument: Comma-separated ``train,val,test``
                fractions.
            standardize_data_splits_argument: Comma-separated ``"y"``/``"n"``
                flags, two per split, controlling feature/target standardization.
            shuffle_data: Whether to randomly shuffle the data before splitting.
            shuffle_seed: Random seed for shuffling; if ``None``, a random seed
                is used.
            deduplicate: ``"none"``, ``"mean"`` or ``"first"``. Duplicate
                feature rows are removed before splitting (avoiding train/test
                leakage); ``"mean"`` averages their targets and ``"first"``
                keeps the first occurrence.

        Returns:
            Tuple of ``((train_features, train_targets),
            (val_features, val_targets), (test_features, test_targets))``, the
            train target statistics ``(y_mean, y_std)``, and the parsed
            standardization flags.

        Raises:
            ValueError: If features or targets are empty.
        """
        split_fractions = helpers.getDatasetSplits(self, split_fractions_argument)
        if len(self.features) == 0 or len(self.targets) == 0:
            raise ValueError(
                f"Cannot split dataset '{self.__class__.__name__}': features or targets are empty"
            )
        rng = np.random.RandomState(shuffle_seed)
        features = self._get_onehot_encoded_features()
        targets = self.targets

        features, targets = self._deduplicate(features, targets, deduplicate)

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

        x_stats, y_stats = self._get_dataset_statistics(
            train_features, train_targets
        )

        standardize_data_splits = helpers.parseStandardizationBools(standardize_data_splits_argument)

        st_train = self._standardize_data(
            train_features,
            train_targets,
            f_means=x_stats.mean,
            f_stds=x_stats.std,
            t_mean=y_stats.mean,
            t_std=y_stats.std,
            standardize_parts=standardize_data_splits.train,
        )
        st_val = self._standardize_data(
            val_features,
            val_targets,
            f_means=x_stats.mean,
            f_stds=x_stats.std,
            t_mean=y_stats.mean,
            t_std=y_stats.std,
            standardize_parts=standardize_data_splits.val,
        )
        st_test = self._standardize_data(
            test_features,
            test_targets,
            f_means=x_stats.mean,
            f_stds=x_stats.std,
            t_mean=y_stats.mean,
            t_std=y_stats.std,
            standardize_parts=standardize_data_splits.test,
        )
        return DatasetData(train_data=st_train, val_data=st_val, test_data=st_test,train_split_target_statistics=y_stats, split_standardizations=standardize_data_splits)


def GetLocal(ds: RegressionDataset) -> tuple[Tensor | None, Tensor | None]:
    """Load a dataset's tensors from the local cache.

    Args:
        ds: Dataset whose ``str()`` name keys the cache file.

    Returns:
        ``(features, targets)`` loaded from disk, or ``(None, None)`` when no
        cached file exists.
    """
    p_str = f"datasets/localfiles/{str(ds)}.pt"
    p = Path(p_str)
    if p.exists():
        tensors = torch.load(p_str)
        features = tensors["features"]
        targets = tensors["targets"]
        return features, targets
    else:
        return None, None


def SaveLocal(ds: RegressionDataset, features: Tensor, targets: Tensor) -> None:
    """Persist a dataset's features and targets to the local cache.

    Args:
        ds: Dataset whose ``str()`` name keys the cache file.
        features: Feature tensor to save.
        targets: Target tensor to save.
    """
    p_str = f"datasets/localfiles/{str(ds)}.pt"
    p = Path(p_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"features": features, "targets": targets}
    torch.save(tensors, p_str)
