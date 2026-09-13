from torch.fft import Tensor
from scaffolds import ResultFileDetails
from datetime import datetime
from scaffolds import RunArguments
from datasets.regression_dataset import RegressionDataset
import os
import subprocess
from pathlib import Path
import json
from sklearn.cluster import KMeans
from torch import Tensor
import torch
import numpy as np


import torch

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def seed_check(seed: int | None, os_scale_training: bool, dset: RegressionDataset, gptype: object, opt_str: str, approx_size: int) -> bool:
    """Check whether a result for this configuration already exists.

    Scans ``results/<dataset>/<model>/*.json`` and compares the seed, output
    scale training flag, optimizer name and approximation size.

    Args:
        seed: Seed to look for.
        os_scale_training: Whether the output scale was trained.
        dset: Dataset, stringified to locate the result directory.
        gptype: Model type, stringified to locate the result directory.
        opt_str: Optimizer display name recorded in results.
        approx_size: Approximation size recorded in results.

    Returns:
        ``True`` if no matching result exists, ``False`` if one was found.
    """
    p = Path(f"results/{str(dset)}/{str(gptype)}")
    tmp = True
    if p.exists():
        for x in os.listdir(p):
            if x.endswith(".json"):
                with open(Path(p, x)) as f:
                    data = json.load(f)
                    if data["seed"] == seed and data["trained_output_scale"] == os_scale_training and data["optimizer"] == opt_str and data.get("approximation_size") == approx_size:
                        tmp = False
    return tmp

def check_repo_clean() -> None:
    """Abort if the git working tree has uncommitted changes.

    Raises:
        RuntimeError: If ``git status --porcelain`` reports any changes.
    """
    # Ensure working tree is clean for reproducibility
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT_DIR,
    ).stdout.strip()
    if git_status:
        msg = (
            "Uncommitted changes detected. Commit or stash them before running "
            "experiments to ensure reproducibility.\n"
            f"Dirty files:\n{git_status}"
        )
        raise RuntimeError(msg)


def get_git_revision_hash() -> str:
    """Return the hash of the current git commit."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()



def getDatasetSplits(dataset: RegressionDataset, split_str: str) -> tuple[float, float, float]:
    """Parse a comma-separated train/val/test split specification.

    Args:
        dataset: Unused; kept for call-site symmetry.
        split_str: Comma-separated ``train,val,test`` fractions.

    Returns:
        Tuple of ``(train, val, test)`` fractions as floats.
    """
    split_str_list = split_str.split(",")


    split_train, split_val, split_test = (
        float(split_str_list[0]),
        float(split_str_list[1]),
        float(split_str_list[2]),
    )
    return split_train, split_val, split_test

def parseStandardizationBools(standardization_str: str) -> list[tuple[bool, bool]]:
    """Parse the standardization flags for the train/val/test splits.

    Args:
        standardization_str: Comma-separated ``"y"``/``"n"`` flags, two per
            split (features, targets), in train, val, test order.

    Returns:
        One ``(standardize_features, standardize_targets)`` tuple per split.
    """
    std_split_str_list = standardization_str.split(",")
    std_split_bool_list = [e == "y" for e in std_split_str_list]
    st_split = [
        (std_split_bool_list[0], std_split_bool_list[1]),
        (std_split_bool_list[2], std_split_bool_list[3]),
        (std_split_bool_list[4], std_split_bool_list[5]),
    ]
    return st_split

def getResultFileDetails(arguments: RunArguments, dataset_name: str, model_name: str, kernel_name: str, optimizer_name: str, start_time_str: str) -> ResultFileDetails:
    """Compute result/log file paths and create their parent directories.

    Args:
        arguments: Run configuration used to build the file name.
        dataset_name: Dataset identifier used as a directory level.
        model_name: Model identifier used as a directory level.
        kernel_name: Kernel display name used in the file name.
        optimizer_name: Optimizer display name used in the file name.
        start_time_str: Timestamp string appended to the file name.

    Returns:
        :class:`ResultFileDetails` with the base name and JSON/CSV paths.
    """
    n = arguments.approximation_size
    seed = arguments.seed
    if arguments.train_signal_variance:
        sig_var_string = "OSTrained"
    else:
        sig_var_string = "OSNotTrained"
    res_path = Path(f"results/{dataset_name}/{model_name}")
    logs_path = Path(res_path, "logs")

    inducing_point_method_str = f"_ipm_{arguments.svgp_strategy}" if arguments.gp == "svgp" else ""
    res_file_name = f"{kernel_name}_{optimizer_name}_{sig_var_string}{inducing_point_method_str}_as{n}_{seed}_{start_time_str}"
    log_file_path = Path(logs_path, f"{res_file_name}.csv")
    res_file_Path= Path(res_path , f"{res_file_name}.json")
    logs_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    return ResultFileDetails(
        baseName=res_file_name,
        logFilePath=log_file_path,
        resultFilePath= res_file_Path
    )


def getInducingPoints(data: Tensor, k: int, strategy: str, seed: int) -> Tensor:
    """Select ``k`` inducing points from ``data``.

    Args:
        data: Training features of shape (n_samples, n_features).
        k: Number of inducing points.
        strategy: ``"kmeans"`` for k-means centroids or ``"random"`` for a
            random subset.
        seed: Random seed for either strategy.

    Returns:
        Tensor of ``k`` inducing points.
    """
    match strategy:
        case "kmeans":
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(data.numpy())
            return torch.tensor(kmeans.cluster_centers_)
        case "random":
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(data))
            return data[idx[:k]]
