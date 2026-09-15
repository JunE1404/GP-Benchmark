from misc.scaffolds import DatasetData
import torch
import gpytorch
from scipy import stats
from torch import Tensor


def evaluate_regression(model: gpytorch.models.GP, predictions: gpytorch.distributions.MultivariateNormal | tuple[Tensor, Tensor], datasetData: DatasetData,picp_levels: tuple[float, ...] = (0.5, 0.9, 0.95)) -> tuple[float, float, dict[float, float], float, float]:
    """Compute regression metrics from predictions and the dataset's test split.

    Predictions and targets are brought back to raw space independently:
    predictions are un-standardized when the *train* targets were standardized
    (that is the space the model was trained in), and targets are un-standardized
    when the *test* targets were standardized. Both then live in raw space.

    Args:
        model: GP model whose lengthscale is reported.
        predictions: Posterior distribution, or a ``(means, stds)`` tuple.
        datasetData: Dataset splits, train target statistics and the
            per-split standardization flags. Test targets come from here.
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

    y_mean = datasetData.train_split_target_statistics.mean.cpu()
    y_std = datasetData.train_split_target_statistics.std.cpu()
    targets = datasetData.test_data.targets

    means = means.cpu()
    targets = targets.cpu()
    stds = stds.cpu()

    if datasetData.split_standardizations.train.targets:
        means = means * y_std + y_mean
        stds = stds * y_std
    if datasetData.split_standardizations.test.targets:
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
