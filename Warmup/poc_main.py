import math

import gpytorch
import numpy as np
import torch
from dataset import Dataset
from model import ExactGPModel


def test_custom_data():

    data_x = torch.linspace(0, 1, 1000)
    data_y = np.sin(data_x * (2 * math.pi)) + torch.randn(data_x.size()) * math.sqrt(
        0.04
    )

    dataset = Dataset()
    dataset.set_data(data_x, data_y)

    (train, test) = dataset.get_split(0.75)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train, test, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.run_training(optimizer, iterations=50)
    model.run_eval(show=True)


def test_uci_data(id):
    dataset = Dataset()
    dataset.load_from_uci(dataset_id=id, dataset_name=None)
    dataset.data.normalize()

    (train, test) = dataset.get_split(0.75)

    print(type(train), type(test))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train, test, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.run_training(optimizer, iterations=250)
    model.run_eval(show=False)


test_uci_data(186)
