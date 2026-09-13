import torch
import gpytorch
from scipy import stats
from torch import Tensor

def evaluate_regression(model: gpytorch.models.GP, predictions: gpytorch.distributions.MultivariateNormal | tuple[Tensor, Tensor], targets: Tensor, y_mean: Tensor | None = None, y_std: Tensor | None = None, targets_standardized: bool = True, trained_output_scale: bool = True, picp_levels: tuple[float, ...] = (0.5, 0.9, 0.95)) -> tuple[float, float, dict[float, float], float, float]:
    """Compute regression metrics from predictions and targets.

    Args:
        model: GP model whose lengthscale is reported.
        predictions: Posterior distribution, or a ``(means, stds)`` tuple.
        targets: Ground-truth target values.
        y_mean: Optional target mean used to invert standardization.
        y_std: Optional target standard deviation used to invert standardization.
        targets_standardized: Whether ``targets`` are in standardized space and
            therefore need un-standardizing alongside the predictions.
        trained_output_scale: Unused; retained for call-site compatibility.
        picp_levels: Coverage levels for which PICP is computed.

    Returns:
        Tuple ``(mae, nll, picp, rmse, lengthscale)`` where ``picp`` maps each
        requested level to its empirical coverage and the remaining values are
        Python floats.
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

    picp = {}
    for alpha in picp_levels:
        z = stats.norm.ppf((1 + alpha) / 2)
        lower = means - z * stds
        upper = means + z * stds
        picp[alpha] = ((targets >= lower) & (targets <= upper)).float().mean().item()

    rmse = torch.sqrt(torch.mean((targets - means) ** 2)).item()

    if hasattr(model.covar_module, 'base_kernel'):
        kernel = model.covar_module.base_kernel
    else:
        kernel = model.covar_module

    l_scale = kernel.lengthscale.norm().item()

    return mae, nll, picp, rmse, l_scale