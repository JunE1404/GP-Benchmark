import gpytorch
from torch import Tensor
from misc.scaffolds import LogDetails
from collections.abc import Callable
from abc import ABC, abstractmethod
import torch

class Regressor(ABC):
    @abstractmethod
    def run_training(self, optimizer: torch.optim.Optimizer, y_mean: Tensor, y_std: Tensor, standardize_test_targets: bool, iterations: int, logger: Callable[[LogDetails], None]) -> float:
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> tuple[gpytorch.distributions.MultivariateNormal, float]:
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
