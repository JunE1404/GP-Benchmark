

from __future__ import annotations

from typing import TYPE_CHECKING

from click import Path
from typing_extensions import List
from typing_extensions import Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from datasets.regression_dataset import RegressionDataset


@dataclass
class RunArguments:
    """Resolved configuration for a single benchmark run."""

    device: str
    dataset: str
    split: str
    standardize: str
    gp: str
    kernel: str
    likelyhood: str
    mean: str
    optimizer: str
    learningrate: float
    lbfgs_max_it: int
    approximation_size: int
    iterations: int
    seed: int
    shuffle: bool
    svgp_strategy: str
    batch_size: int
    train_signal_variance: bool


@dataclass
class EvalGroupArguments:
    """Filters used to select result files for evaluation and plotting."""

    dataset: Optional[List[RegressionDataset]]
    gp: Optional[str] = None
    kernel: Optional[str] = None
    optimizer: Optional[str] = None
    seed: Optional[int] = None
    svgp_strategy: Optional[str] = None
    train_signal_variance: Optional[bool] = None
    n: Optional[int] = None

@dataclass
class WandBDetails:
    """Entity, project and run name for a Weights & Biases run."""

    entity: str
    project: str
    name: str

@dataclass 
class LogDetails:
    """Metrics captured for a single training iteration."""

    iteration: int
    loss: float
    lengthscale: float
    outputscale: float
    likelyhood_noise: float
    test_MAE: float
    test_NLL: float
    test_PICP50: float
    test_PICP90: float
    test_PICP95: float
    test_RMSE: float
    it_time_training: float
    it_time: float


@dataclass 
class RunSummary:
    """Final aggregate test metrics for a run."""

    MAE: float
    NLL: float
    PICP50: float
    PICP90: float
    PICP95: float
    RMSE: float
    training_time: float
    eval_time: float


@dataclass
class ResultFileDetails:
    """Base name and output paths for a run's result and log files."""

    baseName: str
    resultFilePath: Path
    logFilePath: Path