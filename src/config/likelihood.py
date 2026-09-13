from typing_extensions import Tuple
from scaffolds import RunArguments
from  gpytorch.likelihoods import GaussianLikelihood
def getLikelihood(arguments: RunArguments) -> Tuple[GaussianLikelihood | None, str]:
    """Build the likelihood named by ``arguments.likelyhood``.

    Args:
        arguments: Run configuration. ``arguments.likelyhood`` selects the
            likelihood.

    Returns:
        Tuple of ``(likelihood, display_name)``. The likelihood is ``None`` when
        the name matches no supported likelihood.

    Raises:
        ValueError: If ``arguments.likelyhood`` is ``None``.
    """
    if arguments.likelyhood is None:
        raise ValueError("No likelihood argument provided.")
    likelihood = None
    likelihood_name = ""
    match arguments.likelyhood:
        case "gaussian":
            likelihood = GaussianLikelihood()
            likelihood_name = "Gaussian"
    
    return likelihood, likelihood_name