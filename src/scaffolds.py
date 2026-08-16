

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
    val_PICP: float
    val_RMSE: float
    val_pred_mean: float | None = None
    val_pred_median: float | None = None
    val_pred_q05: float | None = None
    val_pred_q95: float | None = None
    val_pred_std_mean: float | None = None


@dataclass 
class RunSummary:
    MAE: float
    NLL: float
    PICP: float
    RMSE: float
    training_time: float
    eval_time: float
