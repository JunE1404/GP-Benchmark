import os
import subprocess

import torch

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_repo_clean():
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def evaluate_regression(model, predictions, targets, y_mean=None, y_std=None, targets_standardized=True):
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
        if targets_standardized:          # <- the gate
            targets = targets * y_std + y_mean

    mae = torch.mean(torch.abs(means - targets)).item()
    nll = -torch.distributions.Normal(means, stds).log_prob(targets).mean().item()

    lower = means - 1.96 * stds
    upper = means + 1.96 * stds
    picp = ((targets >= lower) & (targets <= upper)).float().mean().item()

    rmse = torch.sqrt(torch.mean((targets - means) ** 2)).item()

    l_scale = model.covar_module.base_kernel.lengthscale.norm().item()

    return mae, nll, picp, rmse, l_scale
