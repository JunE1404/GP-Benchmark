import argparse
import importlib
import inspect
import json
import pkgutil
import time
from datetime import datetime
from pathlib import Path

import gpytorch
import torch

import helpers
from datasets.regression_dataset import RegressionDataset
from regressors.cagp import CAGPModel
from regressors.exactgp import ExactGPModel
from regressors.exactgp_conjg_gradients import ExactGPConjGradients
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

    # Force, assume that data is standardized
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


# helpers.check_repo_clean()


parser = argparse.ArgumentParser(
    prog="GP Benchmark",
    description="...",
    epilog="...",
)

parser.add_argument("-f", "--config_file")
parser.add_argument("-dv", "--device")
parser.add_argument("-d", "--datasets")
parser.add_argument("-sp", "--split")
parser.add_argument("-st", "--standardize")
parser.add_argument("-g", "--gp")
parser.add_argument("-k", "--kernel")
parser.add_argument("-l", "--likelyhood")
parser.add_argument("-m", "--mean")
parser.add_argument("-o", "--optimizer")
parser.add_argument("-lr", "--learningrate", type=float)
parser.add_argument("-lit", "--lgbfs_max_it", type=int)
parser.add_argument("-as", "--approximation_size", type=int)
parser.add_argument("-i", "--iterations", type=int)
parser.add_argument("-s", "--seed", type=int)
parser.add_argument("-r", "--shuffle", action="store_true")  # on/off flag

args = parser.parse_args()


sets = instantiate_all_datasets()
print("Loaded datasets:")
print("\n")
for s in sets:
    print(str(s))

split_str_list = args.split.split(",")

split_train, split_val, split_test = (
    float(split_str_list[0]),
    float(split_str_list[1]),
    float(split_str_list[2]),
)
split_fractions = (split_train, split_val, split_test)

std_split_str_list = args.standardize.split(",")
std_split_bool_list = [e == "y" for e in std_split_str_list]
st_split = (
    (std_split_bool_list[0], std_split_bool_list[1]),
    (std_split_bool_list[2], std_split_bool_list[3]),
    (std_split_bool_list[4], std_split_bool_list[5]),
)

gp_select = args.gp

kernel_select = args.kernel


ll_select = args.likelyhood

mean_select = args.mean

op_select = args.optimizer

lr = args.learningrate

if op_select == "lbfgs":
    lbfgs_it = args.lgbfs_max_it
else:
    lbfgs_it = 0

iter = args.iterations
shuffle = args.shuffle

device = args.device
if device != "cuda" and device != "cpu":
    raise ValueError("Invalid device: Use 'cuda' or 'cpu'")

if shuffle:
    seed = args.seed
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
        case "gaussian":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            ll_str = "Gaussian"
        case _:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            ll_str = "Gaussian"

    match kernel_select:
        case "RBF":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            kernel_str = "RBF"
        case "matern2.5":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
            kernel_str = "Matern 2.5"
        case _:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            kernel_str = "RBF"

    match mean_select:
        case "constant":
            mean = gpytorch.means.ConstantMean()
            mean_str = "Constant Mean"
        case _:
            mean = gpytorch.means.ConstantMean()
            mean_str = "Constant Mean"

    match gp_select:
        case "exact":
            model = ExactGPModel(train, test, likelihood, kernel, mean, device)
        case "exactcg":
            model = ExactGPConjGradients(train, test, likelihood, kernel, mean, device)
        case "svgp":
            n = args.approximation_size
            if n > train[0].shape[0]:
                n = train[0].shape[0]
            inducing_points = train[0][:n, :]
            model = SparseVariationalGP(
                inducing_points, train, test, likelihood, kernel, mean, device
            )
        case "cagp":
            model = CAGPModel(
                train,
                test,
                1,
                likelihood,
                kernel=kernel,
                mean_module=mean,
                device=device,
            )
        case _:
            model = ExactGPModel(train, test, likelihood, kernel, mean, device)

    match op_select:
        case 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            opt_str = f"Adam, LR: {lr}"
        case 2:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=lbfgs_it)
            opt_str = f"LBFGS, LR: {lr}, MaxIter: {lbfgs_it}"
        case _:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            opt_str = f"Adam, LR: {lr}"

    time_start = time.time()
    model.run_training(optimizer, iterations=iter)
    time_end = time.time()
    now = datetime.now()
    datetime_str = now.strftime("%d-%m-%Y_%H-%M-%S")
    post = model.predict(test[0])
    eval = {
        "dataset": str(set),
        "modelType": str(model),
        "kernel": kernel_str,
        "likelihood": ll_str,
        "mean": mean_str,
        "optimizer": opt_str,
        "shuffledData": shuffle,
        "seed": seed,
        "evalData": evaluate_regression(post, test[1], y_mean, y_std),
        "trainingTime": time_end - time_start,
        "device": device,
        "git_commit_hash": helpers.get_git_revision_hash(),
        "date": datetime_str,
    }

    results_dir = Path(f"results/{str(set)}/{str(model)}")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{datetime_str}.json", "w") as f:
        json.dump(eval, f, indent=2)
