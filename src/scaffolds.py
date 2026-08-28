

from dataclasses import dataclass


@dataclass
class RunArguments:
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
class WandBDetails:
    entity: str
    project: str
    name: str

@dataclass 
class LogDetails:
    iteration: int
    loss: float
    lengthscale: float
    likelyhood_noise: float
    val_MAE: float
    val_NLL: float
    val_PICP50: float
    val_PICP90: float
    val_PICP95: float
    val_RMSE: float


@dataclass 
class RunSummary:
    MAE: float
    NLL: float
    PICP50: float
    PICP90: float
    PICP95: float
    RMSE: float
    training_time: float
    eval_time: float
