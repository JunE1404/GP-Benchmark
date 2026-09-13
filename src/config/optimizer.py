from typing_extensions import Tuple
import gpytorch
from misc.scaffolds import RunArguments
import torch
def getOptimizer(arguments: RunArguments, model: gpytorch.models.GP)->Tuple[torch.optim.Optimizer, str]:
    """Build the optimizer named by ``arguments.optimizer``.

    Args:
        arguments: Run configuration. ``arguments.optimizer`` selects the
            optimizer and supplies ``learningrate`` / ``lbfgs_max_it``.
        model: Model whose parameters are optimized.

    Returns:
        Tuple of ``(optimizer, display_name)``. The optimizer is ``None`` when
        the name matches no supported optimizer.

    Raises:
        ValueError: If ``arguments.optimizer`` is ``None``.
    """
    if arguments.optimizer is None:
        raise ValueError("No optimizer argument provided.")
    lr = arguments.learningrate
    lbfgs_it = arguments.lbfgs_max_it
    optimizer = None
    optimizer_name = ""
    match arguments.optimizer:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer_name = f"Adam"
        case "lbfgs":
            optimizer = torch.optim.LBFGS(
                model.parameters(),  max_iter=lbfgs_it, line_search_fn="strong_wolfe", max_eval=25
            )
            optimizer_name = f"LBFGS_MaxIter_{lbfgs_it}"

    return optimizer, optimizer_name