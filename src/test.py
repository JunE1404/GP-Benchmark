import math

import gpytorch
import numpy as np
import torch

from Datasets.RegressionDataset import RegressionDataset
from Datasets.UCIParkinsons import UCIParkinsonsTelemonitoring
from Datasets.UCIWine import UCIWineQuality
from Regressors.model import ExactGPModel


def test_uci_data(ds: RegressionDataset):
    dataset = ds
    print(dataset.input_dim)
    split_fractions = (0.7, 0, 0.3)
    st_split_f = (True, True, True)
    st_split_t = (True, True, True)
    (train, val, test) = dataset.get_data_split(
        split_fractions=split_fractions,
        standardize_data_split_features=st_split_f,
        standardize_data_split_targets=st_split_t,
        shuffle_data=True,
        shuffle_seed=100,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    model = ExactGPModel(train, test, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.LBFGS(
    #    model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe"
    # )
    model.run_training(optimizer, iterations=100)
    model.run_eval(show=False)

    print(model.posterior)


def test_custom_data():

    data_x = torch.linspace(0, 1, 1000)
    data_y = np.sin(data_x * (2 * math.pi)) + torch.randn(data_x.size()) * math.sqrt(
        0.04
    )

    dataset = RegressionDataset(data_x, data_y, ["con"])
    split_fractions = (0.7, 0, 0.3)
    st_split_f = (True, True, True)
    st_split_t = (True, True, True)
    (train, val, test) = dataset.get_data_split(
        split_fractions=split_fractions,
        standardize_data_split_features=st_split_f,
        standardize_data_split_targets=st_split_t,
        shuffle_data=True,
        shuffle_seed=100,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train, test, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.LBFGS(
    #    model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe"
    # )
    model.run_training(optimizer, iterations=50)
    model.run_eval(show=True)


test_uci_data(UCIWineQuality())
test_uci_data(UCIParkinsonsTelemonitoring())
test_custom_data()
