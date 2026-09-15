from regressors.regressor import Regressor
from config.kernel import getKernel
from config import likelihood
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime

import misc.helpers as helpers
from misc.scaffolds import RunArguments, WandBDetails, RunSummary
from logging.wab import WandBRun
from config.likelihood import getLikelihood
from config.kernel import getKernel
from config.model import getGPModel
from config.mean import getMean
from config.optimizer import getOptimizer
from config.dataset import getDataset
from config.logging import getLogger
from eval.evaluate_regression import evaluate_regression



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
parser.add_argument("-w", "--wandb", action="store_true")
parser.add_argument("-cl", "--custom_logger", action="store_true")
parser.add_argument("-wp", "--wandb_project", type=str, default="GP Test Runs")
parser.add_argument("-we", "--wandb_entity", type=str, default="GP-Bench-Thesis")

args = parser.parse_args()





def get_from_args() -> RunArguments:
    """Build a :class:`RunArguments` from the parsed command-line arguments.

    Returns:
        The run configuration. Values that are irrelevant for the selected
        optimizer (learning rate for LBFGS) or model are zeroed out.

    Raises:
        ValueError: If the device is neither ``"cuda"`` nor ``"cpu"``.
    """
    split_select = args.split
    gp_select = args.gp
    kernel_select = args.kernel
    app_size = args.approximation_size
    std_select = args.standardize
    ll_select = args.likelyhood
    mean_select = args.mean
    op_select = args.optimizer

    if op_select == "lbfgs":
        lbfgs_it = args.lgbfs_max_it
        lr = 0
    else:
        lbfgs_it = 0
        lr = args.learningrate

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

    wandb_on = args.wandb
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity

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
        train_signal_variance=train_sig_var,
        wandb=wandb_on,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )


def get_from_config(path: str) -> RunArguments:
    """Load a :class:`RunArguments` from a JSON config file.

    Args:
        path: Path to the JSON config file.

    Returns:
        The parsed run configuration.
    """
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
            train_signal_variance=data["trainable_output_scale"],
            wandb=bool(data.get("wandb", False)),
            wandb_project=data.get("wandb_project", "GP Test Runs"),
            wandb_entity=data.get("wandb_entity", "GP-Bench-Thesis"),
        )


def run(arguments: RunArguments) -> None:
    """Run a single benchmark configuration end to end.

    Builds the dataset, splits, likelihood, kernel, mean, model and optimizer,
    trains the model, evaluates it on the test split, and writes the JSON result
    plus the W&B run.

    Args:
        arguments: Resolved run configuration.

    Raises:
        ValueError: If any of the resolved dataset, likelihood, kernel, mean,
            model or optimizer is ``None`` (unknown configuration value).
    """
    dataset = getDataset(arguments)

    if dataset is None:
        raise ValueError(f"Unknown dataset: '{arguments.dataset}'")
    else:

        shuffle = arguments.shuffle
        seed = arguments.seed

        dataset_data = dataset.get_data_splits( 
            split_fractions_argument=arguments.split,
            standardize_data_splits_argument=arguments.standardize,
            shuffle_data=shuffle,
            shuffle_seed=seed,
        )
        train = dataset_data.train_data
        val = dataset_data.val_data
        test = dataset_data.test_data

        device = arguments.device
        lr = arguments.learningrate

        n = arguments.approximation_size
        if n is None or n <= 0:
            n = train.features.shape[0]
        else:
            if n > train.features.shape[0]:
                n = train.features.shape[0]


        likelihood, likelihood_name = getLikelihood(arguments)
        if likelihood is None:
            raise ValueError(f"Unknown likelihood: '{arguments.likelyhood}'")

        kernel, kernel_name = getKernel(arguments, train.features.shape[1])
        if kernel is None:
            raise ValueError(f"Unknown kernel: '{arguments.kernel}'")

        mean, mean_name = getMean(arguments)
        if mean is None:
            raise ValueError(f"Unknown mean: '{arguments.mean}'")

        model = getGPModel(arguments, dataset_data, likelihood, kernel, mean)
        if model is None:
            raise ValueError(f"Unknown GP model: '{arguments.gp}'")
        if not isinstance(model,Regressor):
            raise ValueError(f"Model is not of class Regressor and might not implement needed functions: '{arguments.gp}'")

        optimizer, optimizer_name = getOptimizer(arguments, model)
        if optimizer is None:
            raise ValueError(f"Unknown optimizer: '{arguments.optimizer}'")

        now = datetime.now()
        datetime_str = now.strftime("%d-%m-%Y_%H-%M-%S")

        seed_ok = helpers.seed_check(seed,arguments.train_signal_variance, dataset, model, optimizer_name, n, kernel_name, arguments.svgp_strategy if arguments.gp == "svgp" else None)
        if not seed_ok:
            print(f"Seed {seed} was used already used for {str(dataset)} with {str(model)}")
            return

        result_file_details = helpers.getResultFileDetails(arguments, str(dataset), str(model), kernel_name, optimizer_name, datetime_str)


        if arguments.wandb:
            run_name = f"{str(model)}_{str(dataset)}_{result_file_details.baseName}"
            wandb_details = WandBDetails(entity=arguments.wandb_entity, project=arguments.wandb_project, name=run_name)
            wandb_run = WandBRun(wandb_details, arguments, result_file_details.logFilePath)
            if arguments.custom_logger:
                logger = getLogger()
            else:
                logger = wandb_run.log
        else:
            logger = getLogger()

        training_duration = model.run_training(optimizer, arguments.iterations, logger)
        posterior, fit_time = model.predict(test.features)

        if hasattr(model.covar_module, "outputscale"):
            outputscale_res = model.covar_module.outputscale.item()
        else:
            outputscale_res = 1

        ev_data = evaluate_regression(model, posterior, datasetData=dataset_data)
        
        noise_variance = model.likelihood.noise.item()

        eval = {
            "dataset": str(dataset),
            "approximation_size": n,
            "fulldata": arguments.approximation_size is None,
            "modelType": str(model),
            "inducing_point_method": arguments.svgp_strategy if arguments.gp == "svgp" else None,
            "kernel": kernel_name,
            "trained_output_scale": arguments.train_signal_variance,
            "likelihood": likelihood_name,
            "mean": mean_name,
            "optimizer": optimizer_name,
            "learningrate": lr,
            "shuffledData": shuffle,
            "seed": seed,
            "evalData": {"MAE": ev_data[0], 
                         "NLL":ev_data[1], 
                         "PICP50":ev_data[2][0.5],
                         "PICP90":ev_data[2][0.9],
                         "PICP95":ev_data[2][0.95], 
                         "RMSE":ev_data[3], 
                         "Lengthscale":ev_data[4], 
                         "Output_scale": outputscale_res,
                         "Noise_variance": noise_variance},
            "trainingTime": training_duration,
            "evalTime": fit_time,
            "device": device,
            "git_commit_hash": helpers.get_git_revision_hash(),
            "date": datetime_str,
        }

        if arguments.wandb:
            summary = RunSummary(MAE=ev_data[0],
                                 NLL=ev_data[1], 
                                 PICP50=ev_data[2][0.5],
                                 PICP90=ev_data[2][0.9],
                                 PICP95=ev_data[2][0.95], 
                                 RMSE=ev_data[3], 
                                 training_time=training_duration, 
                                 eval_time=fit_time)

            wandb_run.summarize(summary)
            wandb_run.finish()

        with open(result_file_details.resultFilePath, "w") as f:
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
            try:
                run(arguments)
            except Exception as e:
                print("Training of "+ path+ " failed")
        else:
            pass
else:
    arguments = get_from_args()
    run(arguments)
