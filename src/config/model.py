import gpytorch
from torch.fft import Tensor
from misc.scaffolds import RunArguments
from regressors.cagp import CAGPModel
from regressors.exactgp import ExactGPModel
from regressors.exactgp_conjg_gradients import ExactGPCGModel
from regressors.svgp import SparseVariationalGP
def getGPModel(arguments: RunArguments, train: Tensor, val: Tensor, test: Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel: gpytorch.kernels.Kernel, mean: gpytorch.means.Mean) -> gpytorch.models.GP|None:
    """Instantiate the GP model named by ``arguments.gp``.

    Args:
        arguments: Run configuration. ``arguments.gp`` selects the model class
            and supplies ``approximation_size``, ``device``, ``seed``,
            ``svgp_strategy`` and ``batch_size``.
        train: Training split as ``(features, targets)``.
        val: Validation split as ``(features, targets)``.
        test: Test split as ``(features, targets)``.
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
    device = arguments.device
    seed = arguments.seed
    model = None
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
    return model
