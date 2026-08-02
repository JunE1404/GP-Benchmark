

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
