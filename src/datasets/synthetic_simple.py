import math

import numpy as np
import torch

from .regression_dataset import RegressionDataset


class SimpleSyntheticDataset(RegressionDataset):
    def __init__(self) -> None:
        """Create a 1-D synthetic sinusoid dataset with Gaussian noise.

        Generates 1000 equally spaced inputs on ``[0, 1]`` and targets from
        ``sin(2 * pi * x)`` corrupted by Gaussian noise with variance 0.04.
        """
        data_x = torch.linspace(0, 1, 1000).unsqueeze(-1)
        data_y = np.sin(data_x * (2 * math.pi)) + torch.randn(
            data_x.size()
        ) * math.sqrt(0.04)
        feature_types = ["con"]
        super().__init__(data_x, data_y, feature_types)
