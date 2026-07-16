import argparse
import importlib
import inspect
import json
import pkgutil
import time
import os
from datetime import datetime
from pathlib import Path

import gpytorch
import torch

import helpers
from dataclasses import dataclass
from datasets.regression_dataset import RegressionDataset
from regressors.cagp import CAGPModel
from regressors.exactgp import ExactGPModel
from regressors.exactgp_conjg_gradients import ExactGPCGModel
from regressors.svgp import SparseVariationalGP
from datasets.synthetic_simple import SimpleSyntheticDataset
from datasets.uci_parkinsons import UCIParkinsonsTelemonitoring
from datasets.uci_wine import UCIWineQuality

from gpytorch.kernels.keops import RBFKernel as RBFKEops
from gpytorch.kernels.keops import MaternKernel as MaternKeops

from kmeans import getInducingPoints


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

parser.add_argument("-f", "--config")
parser.add_argument("-dv", "--device")
parser.add_argument("-d", "--dataset")
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




def get_from_args() -> RunArguments:
    split_select = args.split

    gp_select = args.gp

    kernel_select = args.kernel

    app_size = args.approximation_size

    std_select = args.standardize

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


    s = args.dataset

    return RunArguments(approximation_size=app_size,
                        dataset=s,
                        device=device,
                        gp=gp_select,
                        iterations=iter,
                        kernel=kernel_select,
                        learningrate=lr,
                        likelyhood=ll_select,
                        mean=mean_select,
                        shuffle=shuffle,
                        seed=seed,
                        lbfgs_max_it=lbfgs_it,
                        split=split_select,
                        optimizer=op_select,
                        standardize=std_select)

def get_from_config(path: str):
    with open(path,"r") as f:
        data = json.load(f)
        if data["optimizer"] == "lbfgs":
            lbfgs_max_it = data["lbfgs_max_iter"]
        else:
            lbfgs_max_it = None

        return RunArguments(approximation_size=data["approximation_size"],
                            dataset=data["dataset"],
                            device=data["device"],
                            gp=data["gp"],
                            iterations=int(data["iterations"]),
                            kernel=data["kernel"],
                            learningrate=data["learningrate"],
                            likelyhood=data["likelyhood"],
                            mean=data["mean"],
                            shuffle=bool(data["shuffle"]),
                            seed=int(data["seed"]),
                            lbfgs_max_it=lbfgs_max_it,
                            split=data["data_split"],
                            optimizer=data["optimizer"],
                            standardize=data["data_standartization"])

def run(arguments: RunArguments):
    match arguments.dataset:
        case "synth":
            dset = SimpleSyntheticDataset()
        case "parkinsons":
            dset = UCIParkinsonsTelemonitoring()
        case "wine":
            dset = UCIWineQuality()
        case _:
            dset = None

    if dset is not None:

        split_str_list = arguments.split.split(",")

        iter = arguments.iterations

        split_train, split_val, split_test = (
            float(split_str_list[0]),
            float(split_str_list[1]),
            float(split_str_list[2]),
        )
        split_fractions = (split_train, split_val, split_test)

        std_split_str_list = arguments.standardize.split(",")
        std_split_bool_list = [e == "y" for e in std_split_str_list]
        st_split = (
            (std_split_bool_list[0], std_split_bool_list[1]),
            (std_split_bool_list[2], std_split_bool_list[3]),
            (std_split_bool_list[4], std_split_bool_list[5]),
        )

        shuffle = arguments.shuffle
        seed = arguments.seed

        (train, val, test), (y_mean, y_std) = dset.get_data_split(
            split_fractions=split_fractions,
            standardize_data_splits=st_split,
            shuffle_data=shuffle,
            shuffle_seed=seed,
        )

        device = arguments.device

        lr = arguments.learningrate
        lbfgs_it = arguments.lbfgs_max_it

        n = arguments.approximation_size
        if n is None:
            n = train[0].shape[0]
        else:
            if n > train[0].shape[0]:
                n = train[0].shape[0]

        match arguments.likelyhood:
            case "gaussian":
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"
            case _:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"

        match arguments.kernel:
            case "RBF":
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                kernel_str = "RBF"
            case "matern2.5":
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
                kernel_str = "Matern 2.5"
            case "RBFKeops":
                kernel = gpytorch.kernels.ScaleKernel(RBFKEops())
                kernel_str = "RBF Keops"
            case "matern2.5Keops":
                kernel = gpytorch.kernels.ScaleKernel(MaternKeops(nu=2.5))
                kernel_str = "Matern 2.5 Keops"
            case _:
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                kernel_str = "RBF"

        match arguments.mean:
            case "constant":
                mean = gpytorch.means.ConstantMean()
                mean_str = "Constant Mean"
            case _:
                mean = gpytorch.means.ConstantMean()
                mean_str = "Constant Mean"

        match arguments.gp:
            case "exact":
                train_points = train[0][:n,:], train[1][:n]
                model = ExactGPModel(train_points, test, likelihood, kernel, mean, device)
            case "exactcg":
                train_points = train[0][:n,:], train[1][:n]
                model = ExactGPCGModel(train_points, test, likelihood, kernel, mean, device)
            case "svgp":
                if n == train[0].shape[0]:
                    inducing_points = train
                else:
                    inducing_points = getInducingPoints(train[0], n)
                model = SparseVariationalGP(
                    inducing_points, train, test, likelihood, kernel, mean, device
                )
            case "cagp":
                model = CAGPModel(
                    train,
                    test,
                    n,
                    likelihood,
                    kernel=kernel,
                    mean_module=mean,
                    device=device,
                )
            case _:
                model = ExactGPModel(train, test, likelihood, kernel, mean, device)

        match arguments.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                opt_str = f"Adam, LR: {lr}"
            case "lbfgs":
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
        start_time_eval = time.time()
        post = model.predict(test[0])
        end_time_eval = time.time()
        eval = {
            "dataset": str(dset),
            "approximation_size": n,
            "modelType": str(model),
            "kernel": kernel_str,
            "likelihood": ll_str,
            "mean": mean_str,
            "optimizer": opt_str,
            "shuffledData": shuffle,
            "seed": seed,
            "evalData": evaluate_regression(post, test[1], y_mean, y_std),
            "trainingTime": time_end - time_start,
            "evalTime": end_time_eval - start_time_eval,
            "device": device,
            "git_commit_hash": helpers.get_git_revision_hash(),
            "date": datetime_str,
        }

        results_dir = Path(f"results/{str(dset)}/{str(model)}")
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"{datetime_str}.json", "w") as f:
            json.dump(eval, f, indent=2)


if args.config is not None:
    path = args.config
    if os.path.exists(path):
        if os.path.isdir(path):
            dir_list = os.listdir(path)
            for p in dir_list:
                path_full = path+"/"+p
                arguments = get_from_config(path_full)
                run(arguments)
        elif os.path.isfile(path):
            arguments = get_from_config(path)
            run(arguments)
        else:
            pass
else:
    arguments = get_from_args()
    run(arguments)
