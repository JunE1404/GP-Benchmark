import importlib
import inspect
import json
import math
import os
import pkgutil
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch

from datasets.regression_dataset import RegressionDataset
from regressors.exactgp import ExactGPModel
from regressors.svgp import SparseVariationalGP


def instantiate_all_datasets():
    """Find and instantiate every concrete dataset class in the datasets package."""
    datasets = []
    for importer, modname, is_pkg in pkgutil.iter_modules(["datasets"]):
        module = importlib.import_module(f"datasets.{modname}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, RegressionDataset)
                and cls.__init__ is not RegressionDataset.__init__
            ):
                datasets.append(cls())

    return datasets


def evaluate_regression(predictions, targets, y_mean=None, y_std=None):
    """Compute regression metrics from predictions and targets.

    Args:
        predictions: Distribution or tuple (means, stds).
        targets: Ground-truth target values.
        y_mean, y_std: Optional standardization stats to invert.

    Returns:
        dict with MAE, NLL, PICP, RMSE (as Python floats).
    """
    if hasattr(predictions, "mean"):
        means, stds = predictions.mean, predictions.stddev
    else:
        means, stds = predictions

    means = means.cpu()
    targets = targets.cpu()
    stds = stds.cpu()

    # Optional: revert standardization to original space
    if y_mean is not None and y_std is not None:
        means = means * y_std + y_mean
        stds = stds * y_std
        targets = targets * y_std + y_mean

    mae = torch.mean(torch.abs(means - targets)).item()
    nll = -torch.distributions.Normal(means, stds).log_prob(targets).mean().item()

    lower = means - 1.96 * stds
    upper = means + 1.96 * stds
    picp = ((targets >= lower) & (targets <= upper)).float().mean().item()

    rmse = torch.sqrt(torch.mean((targets - means) ** 2)).item()

    return {"MAE": mae, "NLL": nll, "PICP": picp, "RMSE": rmse}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    print("Training and evaluation script")
    print("Loading datasets...")

    sets = instantiate_all_datasets()
    print("Loaded datasets:")
    print("\n")
    for s in sets:
        print(str(s))

    print("\n")
    print("Choose dataset split: [training], [validation], [testing]")
    split_str_list = input().split(",")

    split_train, split_val, split_test = (
        float(split_str_list[0]),
        float(split_str_list[1]),
        float(split_str_list[2]),
    )
    split_fractions = (split_train, split_val, split_test)
    print("\n")
    print(
        "Standardize training_x, training_y, validation_x, validation_y, testing_x, testing_y? (y/n)"
    )
    std_split_str_list = input().split(",")
    std_split_bool_list = [e == "y" for e in std_split_str_list]
    st_split = (
        (std_split_bool_list[0], std_split_bool_list[1]),
        (std_split_bool_list[2], std_split_bool_list[3]),
        (std_split_bool_list[4], std_split_bool_list[5]),
    )
    print("\n")
    print("Select GP type:")
    print("1. Exact")
    print("2. SVGP")

    gp_select = int(input())

    print("\n")
    print("Select kernel:")
    print("1. RBF")
    print("2. Matern 2.5")

    kernel_select = int(input())

    print("\n")

    print("Select Likelihood type:")
    print("1. Gaussian")

    ll_select = int(input())

    print("\n")

    print("Select Optimizer type:")
    print("1. Adam")
    print("2. LBFGS")

    op_select = int(input())

    print("\n")

    print("Choose optimizer learning rate:")
    lr = float(input())

    print("\n")

    if op_select == 2:
        print("Choose LBFGS max iteration count:")
        lbfgs_it = int(input())
        print("\n")
    else:
        lbfgs_it = 0

    print("Choose training loop iteration count:")
    iter = int(input())
    print("\n")

    print("Shuffle data? (y/n)")
    shuffle = input() == "y"

    print("\n")

    if shuffle:
        print("Input seed:")
        seed = int(input())
    else:
        seed = None

    for set in sets:
        (train, val, test), (y_mean, y_std) = set.get_data_split(
            split_fractions=split_fractions,
            standardize_data_splits=st_split,
            shuffle_data=shuffle,
            shuffle_seed=seed,
        )

        match ll_select:
            case 1:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"
            case _:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"

        match kernel_select:
            case 1:
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                kernel_str = "RBF"
            case 2:
                kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5)
                )
                kernel_str = "Matern 2.5"
            case _:
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                kernel_str = "RBF"

        match gp_select:
            case 1:
                model = ExactGPModel(train, test, likelihood, kernel)
            case 2:
                n = train[0].size(dim=0)
                n = math.ceil(n / 20)
                inducing_points = train[0][:n, :]
                model = SparseVariationalGP(inducing_points, train, test, likelihood)

            case _:
                model = ExactGPModel(train, test, likelihood, kernel)

        match op_select:
            case 1:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                opt_str = f"Adam, LR: {lr}"
            case 2:
                optimizer = torch.optim.LBFGS(
                    model.parameters(), lr=lr, max_iter=lbfgs_it
                )
                opt_str = f"LBFGS, LR: {lr}, MaxIter: {lbfgs_it}"
            case _:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                opt_str = f"Adam, LR: {lr}"

        time_start = time.time()
        model.run_training(optimizer, iterations=iter)
        time_end = time.time()

        post = model.predict(test[0])
        eval = {
            "dataset": str(set),
            "modelType": str(model),
            "kernel": kernel_str,
            "likelihood": ll_str,
            "optimizer": opt_str,
            "shuffledData": shuffle,
            "seed": seed,
            "evalData": evaluate_regression(post, test[1], y_mean, y_std),
            "trainingTime": time_end - time_start,
        }

        results_dir = Path(f"results/{str(set)}/{str(model)}")
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "results.json", "w") as f:
            json.dump(eval, f, indent=2)


main()
