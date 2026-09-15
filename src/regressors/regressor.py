
from typing_extensions import Callable
from typing import Any
from misc.scaffolds import DatasetData
import gpytorch
from torch import Tensor
from abc import ABC, abstractmethod
import torch


class Regressor(ABC):
    """Minimal contract every regressor must satisfy.

    The only enforced interface is ``run_training(optimizer, iterations)``.
    Everything else -- the constructor signature, extra training arguments
    (loggers, statistics, ...), hyperparameters -- is up to the implementation.
    """

    datasetData: DatasetData
    @abstractmethod
    def __init__(
        self,
        dataset: DatasetData,
        n: int,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        device: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    @abstractmethod
    def run_training(self, optimizer: torch.optim.Optimizer, iterations: int, logger:  Callable[[Any], None]) -> float:
        """Train the model and return its wall-clock duration."""
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> tuple[gpytorch.distributions.MultivariateNormal, float]:
        """Return the posterior at ``x`` and the (negative) prediction duration."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the display name used in result file paths."""
        pass
