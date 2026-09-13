from typing_extensions import Tuple
import gpytorch
from misc.scaffolds import RunArguments
from gpytorch.means import ConstantMean
def getMean(arguments: RunArguments)-> Tuple[gpytorch.means.Mean| None, str]:
    """Build the mean module named by ``arguments.mean``.

    Args:
        arguments: Run configuration. ``arguments.mean`` selects the mean
            function.

    Returns:
        Tuple of ``(mean, display_name)``. The mean is ``None`` when the name
        matches no supported mean.

    Raises:
        ValueError: If ``arguments.mean`` is ``None``.
    """
    if arguments.mean is None:
        raise ValueError("No mean argument provided.")
    mean = None
    mean_name = ""
    match arguments.mean:
        case "constant":
            mean = ConstantMean()
            mean_name = "Constant Mean"
    return mean, mean_name