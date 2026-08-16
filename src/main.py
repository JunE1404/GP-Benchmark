import argparse
import importlib
import inspect
import json
import os
import pkgutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gpytorch
import torch
from gpytorch.kernels.keops import MaternKernel as MaternKeops
from gpytorch.kernels.keops import RBFKernel as RBFKEops

import helpers
from datasets.regression_dataset import RegressionDataset
from datasets.synthetic_simple import SimpleSyntheticDataset
from datasets.uci_parkinsons import UCIParkinsonsTelemonitoring
from datasets.uci_proteins import UCIProtein
from datasets.uci_wine import UCIWineQuality
from kmeans import getInducingPoints
from regressors.cagp import CAGPModel
from regressors.exactgp import ExactGPModel
from regressors.exactgp_conjg_gradients import ExactGPCGModel
from regressors.svgp import SparseVariationalGP
from scaffolds import RunArguments, WandBDetails, RunSummary
from wab import WandBRun


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
parser.add_argument("-r", "--shuffle", action="store_true") 
parser.add_argument("-bs", "--svgp_batch_size", type=int)
parser.add_argument("-str", "--svgp_strategy", type=str)
parser.add_argument("-os", "--trainable_output_scale", action="store_true")

args = parser.parse_args()


def seed_check(seed,os_scale_training, dset, gptype):
    p = Path(f"results/{str(dset)}/{str(gptype)}")
    tmp = True
    if p.exists():
        for x in os.listdir(p):
            if x.endswith(".json"):
                with open(Path(p, x)) as f:
                    data = json.load(f)
                    if data["seed"] == seed and data["trained_output_scale"] == os_scale_training:
                        tmp = False
    return tmp



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

    svgp_strat = args.svgp_strategy
    batch_size = args.svgp_batch_size
    train_sig_var = args.trainable_output_scale

    return RunArguments(
        approximation_size=app_size,
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
        standardize=std_select,
        svgp_strategy=svgp_strat,
        batch_size=batch_size,
        train_signal_variance=train_sig_var
    )


def get_from_config(path: str):
    with open(path, "r") as f:
        data = json.load(f)
        if data["optimizer"] == "lbfgs":
            lbfgs_max_it = data["lbfgs_max_iter"]
            learningrate = 0
        else:
            lbfgs_max_it = None
            learningrate=data["learningrate"]

        if data["gp"] == "svgp":
            svgp_strategy=data["svgp_strategy"]
            batch_size=data["svgp_batch_size"]
        else:
            svgp_strategy=""
            batch_size=0

        return RunArguments(
            approximation_size=data["approximation_size"],
            dataset=data["dataset"],
            device=data["device"],
            gp=data["gp"],
            iterations=int(data["iterations"]),
            kernel=data["kernel"],
            learningrate=learningrate,
            likelyhood=data["likelyhood"],
            mean=data["mean"],
            shuffle=bool(data["shuffle"]),
            seed=int(data["seed"]),
            lbfgs_max_it=lbfgs_max_it,
            split=data["data_split"],
            optimizer=data["optimizer"],
            standardize=data["data_standartization"],
            svgp_strategy=svgp_strategy,
            batch_size=batch_size,
            train_signal_variance=data["trainable_output_scale"]
        )


def run(arguments: RunArguments):
    match arguments.dataset:
        case "synth":
            dset = SimpleSyntheticDataset()
        case "parkinsons":
            dset = UCIParkinsonsTelemonitoring()
        case "wine":
            dset = UCIWineQuality()
        case "protein":
            dset = UCIProtein()
        case _:
            dset = None
    print(f"Dataset: {str(dset)}")

    if dset is not None:
        split_str_list = arguments.split.split(",")

        iter = arguments.iterations

        split_train, split_val, split_test = (
            float(split_str_list[0]),
            float(split_str_list[1]),
            float(split_str_list[2]),
        )
        split_fractions = (split_train, split_val, split_test)
        print(split_fractions)

        std_split_str_list = arguments.standardize.split(",")
        std_split_bool_list = [e == "y" for e in std_split_str_list]
        st_split = (
            (std_split_bool_list[0], std_split_bool_list[1]),
            (std_split_bool_list[2], std_split_bool_list[3]),
            (std_split_bool_list[4], std_split_bool_list[5]),
        )

        standardize_val_targets = st_split[1][1]
        standardize_test_targets = st_split[2][1]

        shuffle = arguments.shuffle
        seed = arguments.seed

        print(f"Shuffle data?: {shuffle}, Seed: {seed}")

        (train, val, test), (y_mean, y_std) = dset.get_data_split( #always standard features, dont stand targets for val and test, but report metrics in normal space
            split_fractions=split_fractions,
            standardize_data_splits=st_split,
            shuffle_data=shuffle,
            shuffle_seed=seed,
        )

        device = arguments.device
        print(f"Device: {device}")

        lr = arguments.learningrate
        lbfgs_it = arguments.lbfgs_max_it

        print(f"Learningrate: {lr}")

        n = arguments.approximation_size
        if n is None:
            n = train[0].shape[0]
        else:
            if n > train[0].shape[0]:
                n = train[0].shape[0]

        print(f"Approximation size: {n}")

        match arguments.likelyhood:
            case "gaussian":
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"
            case _:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                ll_str = "Gaussian"

        print(f"Likelihood: {ll_str}")

        def kernelWrap(k: gpytorch.kernel.Kernel):
            if arguments.train_signal_variance:
                return gpytorch.kernels.ScaleKernel(k)
            else:
                return k

        match arguments.kernel:
            case "RBF":
                # to fix output scale, dont wrap in scale kernel, make adj via "trainable_output_scale parameter"
                kernel = kernelWrap(gpytorch.kernels.RBFKernel(
                        ard_num_dims=train[0].shape[1],
                        lengthscale_constraint=gpytorch.constraints.GreaterThan(10e-6),
                    ))
                kernel_str = "RBF"
            case "matern2.5":
                kernel = kernelWrap(
                    gpytorch.kernels.MaternKernel(nu=2.5)
                )
                kernel_str = "Matern 2.5"
            case "RBFKeops":
                kernel = kernelWrap(
                    RBFKEops(
                        ard_num_dims=train[0].shape[1],
                        lengthscale_constraint=gpytorch.constraints.GreaterThan(10e-6),
                    )
                )
                kernel_str = "RBF Keops"
            case "matern2.5Keops":
                kernel = kernelWrap(MaternKeops(nu=2.5))
                kernel_str = "Matern 2.5 Keops"
            case _:
                kernel = kernelWrap(gpytorch.kernels.RBFKernel())
                kernel_str = "RBF"

        str_vartrained = ", Trainable output scale" if (arguments.train_signal_variance) else ""
        print(f"Kernel: {kernel_str}{str_vartrained}")
        train_sig_var = arguments.train_signal_variance

        match arguments.mean:
            case "constant":
                mean = gpytorch.means.ConstantMean()
                mean_str = "Constant Mean"
            case _:
                mean = gpytorch.means.ConstantMean()
                mean_str = "Constant Mean"

        print(f"Mean: {mean_str}")

        match arguments.gp:
            case "exact":
                train_points = train[0][:n, :], train[1][:n]
                model = ExactGPModel(
                    train_points, test, val, likelihood, kernel, mean, device
                )
            case "exactcg":
                train_points = train[0][:n, :], train[1][:n]
                model = ExactGPCGModel(
                    train_points, test, val, likelihood, kernel, mean, device
                )
            case "svgp":
                model = SparseVariationalGP(
                    arguments.svgp_strategy,seed, n, train, test, val,arguments.batch_size, likelihood, kernel, mean, device
                )
            case "cagp":
                model = CAGPModel(
                    train,
                    test,
                    val,
                    n,
                    likelihood,
                    kernel=kernel,
                    mean_module=mean,
                    device=device,
                )
            case _:
                model = ExactGPModel(train, test, val, likelihood, kernel, mean, device)

        print(f"GP Model: {str(model)}")

        match arguments.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                opt_str = f"Adam"
            case "lbfgs":
                optimizer = torch.optim.LBFGS(
                    model.parameters(),  max_iter=lbfgs_it, line_search_fn="strong_wolfe"
                )
                opt_str = f"LBFGS, MaxIter: {lbfgs_it}"
            case _:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                opt_str = f"Adam"

        print(f"Optimizer: {opt_str}")

        now = datetime.now()
        datetime_str = now.strftime("%d-%m-%Y_%H-%M-%S")

        seed_ok = seed_check(seed,train_sig_var, dset, model)
        if not seed_ok:
            print(f"Seed {seed} was used already used for {str(dset)} with {str(model)}")
            return

        if train_sig_var:
            sig_var_string = "OSTrained"
        else:
            sig_var_string = "OSNotTrained"
        res_path = Path(f"results/{str(dset)}/{str(model)}")
        log_path = Path(res_path, "logs")
        res_file_name = f"{kernel_str}_{opt_str}_{sig_var_string}_{seed}_{datetime_str}"
        log_file_path = Path(log_path, f"{res_file_name}.csv")
        log_path.mkdir(parents=True, exist_ok=True)
        res_path.mkdir(parents=True, exist_ok=True)

        run_name = f"{str(model)}_{str(dset)}_{kernel_str}_{opt_str}_{sig_var_string}_{str(seed)}_{datetime_str}"
        wandb_details = WandBDetails(entity="GP-Bench-Thesis", project="GP Test Runs", name=run_name)
        wandb_run = WandBRun(wandb_details, arguments, log_file_path)
        logger = wandb_run.log

        time_start = time.time()
        model.run_training(optimizer,y_mean, y_std,standardize_val_targets, iterations=iter, logger=logger)
        time_end = time.time()
        start_time_eval = time.time()
        post = model.predict(test[0])
        end_time_eval = time.time()
        ev_data = helpers.evaluate_regression(model,post, test[1], y_mean, y_std, standardize_test_targets, train_sig_var)
        t_time = time_end - time_start
        e_time = end_time_eval - start_time_eval
        eval = {
            "dataset": str(dset),
            "approximation_size": n,
            "modelType": str(model),
            "kernel": kernel_str,
            "trained_output_scale": train_sig_var,
            "likelihood": ll_str,
            "mean": mean_str,
            "optimizer": opt_str,
            "learningrate": lr,
            "shuffledData": shuffle,
            "seed": seed,
            "evalData": {"MAE": ev_data[0], "NLL":ev_data[1], "PICP":ev_data[2], "RMSE":ev_data[3], "Lengthscale":ev_data[4]},
            "trainingTime": t_time,
            "evalTime": e_time,
            "device": device,
            "git_commit_hash": helpers.get_git_revision_hash(),
            "date": datetime_str,
        }

        summary = RunSummary(MAE=ev_data[0], NLL=ev_data[1], PICP=ev_data[2], RMSE=ev_data[3], training_time=t_time, eval_time=e_time)
        
        wandb_run.summarize(summary)
        wandb_run.finish()
        with open(Path(res_path , f"{res_file_name}.json"), "w") as f:
            json.dump(eval, f, indent=2)


if args.config is not None:
    path = args.config
    print(path)
    if os.path.exists(path):
        if os.path.isdir(path):
            dir_list = os.listdir(path)
            for p in dir_list:
                path_full = path + "/" + p
                arguments = get_from_config(path_full)
                try:
                    run(arguments)
                except:
                    print("Training of "+ path+ " failed")
        elif os.path.isfile(path):
            arguments = get_from_config(path)
            run(arguments)
            #try:
            #    run(arguments)
            #except Exception as e:
            #    print("Training of "+ path+ " failed")
            #    print(repr(e)) 
        else:
            pass
else:
    arguments = get_from_args()
    run(arguments)
