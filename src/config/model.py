from dataclasses import replace

import gpytorch
from misc.scaffolds import DatasetData, RunArguments, Split
from regressors.cagp import CAGPModel
from regressors.exactgp import ExactGPModel
from regressors.exactgp_conjg_gradients import ExactGPCGModel
from regressors.svgp import SparseVariationalGP


def getGPModel(arguments: RunArguments, dataset: DatasetData, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel: gpytorch.kernels.Kernel, mean: gpytorch.means.Mean) -> gpytorch.models.GP | None:
    """Instantiate the GP model named by ``arguments.gp``.

    Args:
        arguments: Run configuration. ``arguments.gp`` selects the model class
            and supplies ``approximation_size``, ``device``, ``seed``,
            ``svgp_strategy`` and ``batch_size``.
        dataset: Standardized train/val/test splits plus the train target
            statistics and standardization flags.
        likelihood: Likelihood passed to the model.
        kernel: Covariance kernel passed to the model.
        mean: Mean module passed to the model.

    Returns:
        The constructed GP model, or ``None`` when the name matches no supported
        model.

    Raises:
        ValueError: If ``arguments.gp`` is ``None``.
    """
    if arguments.gp is None:
        raise ValueError("No GP model argument provided.")
    n = arguments.approximation_size
    train_size = dataset.train_data.features.shape[0]
    if n is None or n <= 0:
        n = train_size
    else:
        n = min(n, train_size)
    device = arguments.device
    seed = arguments.seed
    model = None
    match arguments.gp:
        case "exact":
            truncated = replace(
                dataset,
                train_data=Split(
                    dataset.train_data.features[:n, :], dataset.train_data.targets[:n]
                ),
            )
            model = ExactGPModel(truncated, likelihood, kernel, mean, device)
        case "exactcg":
            truncated = replace(
                dataset,
                train_data=Split(
                    dataset.train_data.features[:n, :], dataset.train_data.targets[:n]
                ),
            )
            model = ExactGPCGModel(truncated, likelihood, kernel, mean, device)
        case "svgp":
            model = SparseVariationalGP(
                dataset,
                arguments.svgp_strategy,
                seed,
                n,
                arguments.batch_size,
                likelihood,
                kernel,
                mean,
                device,
            )
        case "cagp":
            model = CAGPModel(
                dataset,
                projection_dim=n,
                likelihood=likelihood,
                kernel=kernel,
                mean_module=mean,
                device=device,
            )
    return model
