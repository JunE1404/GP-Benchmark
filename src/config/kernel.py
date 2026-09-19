from typing_extensions import Tuple
from gpytorch.kernels.keops import MaternKernel as MaternKeops
from gpytorch.kernels.keops import RBFKernel as RBFKEops
import gpytorch

from misc.scaffolds import RunArguments

def signal_variance_kernelWrap(
    train_sig_var: bool, k: gpytorch.kernels.Kernel
) -> gpytorch.kernels.Kernel:
    """Optionally wrap a kernel so that its output scale is trainable.

    Args:
        train_sig_var: When ``True``, wrap ``k`` in a ``ScaleKernel``.
        k: Base covariance kernel.

    Returns:
        ``ScaleKernel(k)`` if ``train_sig_var`` is ``True``, otherwise ``k``.
    """
    if train_sig_var:
        return gpytorch.kernels.ScaleKernel(k)
    else:
        return k


def _parse_lengthscale_constraint(spec: str | None) -> gpytorch.constraints.Constraint | None:
    """Build a lengthscale constraint from a ``"lo"`` or ``"lo,hi"`` spec.

    ``"lo"`` yields a lower bound (``GreaterThan``) and preserves the default
    initial lengthscale; ``"lo,hi"`` yields an ``Interval`` (note that this
    changes the initial lengthscale to the interval midpoint). ``None`` yields
    ``None`` so the kernel keeps its default constraint untouched.

    Args:
        spec: Bound specification or ``None``.

    Returns:
        The constraint, or ``None`` when ``spec`` is ``None``.

    Raises:
        ValueError: If the spec is malformed or the bounds are invalid.
    """
    if spec is None:
        return None
    try:
        parts = [float(part) for part in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid lengthscale bounds '{spec}': expected 'lo' or 'lo,hi'") from exc
    if len(parts) == 1:
        if parts[0] <= 0:
            raise ValueError(f"Invalid lengthscale lower bound '{spec}': must be > 0")
        return gpytorch.constraints.GreaterThan(parts[0])
    if len(parts) == 2:
        lower, upper = parts
        if not 0 < lower < upper:
            raise ValueError(f"Invalid lengthscale bounds '{spec}': expected 0 < lo < hi")
        return gpytorch.constraints.Interval(lower, upper)
    raise ValueError(f"Invalid lengthscale bounds '{spec}': expected 'lo' or 'lo,hi'")


def getKernel(
    arguments: RunArguments, number_train_points: int
) -> Tuple[gpytorch.kernels.Kernel | None, str]:
    """Build the covariance kernel named by ``arguments.kernel``.

    Args:
        arguments: Run configuration. ``arguments.kernel`` selects the kernel and
            ``arguments.lengthscale_bounds`` optionally constrains the
            lengthscale (``"lo"`` for a lower bound, ``"lo,hi"`` for an
            interval). When it is ``None`` no constraint argument is passed, so
            the kernel keeps its default constraint and unbounded runs are
            unaffected.
        number_train_points: Number of input dimensions, used for ARD kernels.

    Returns:
        Tuple of ``(kernel, display_name)``. The kernel is ``None`` when the name
        matches no supported kernel.

    Raises:
        ValueError: If ``arguments.kernel`` is ``None``.
    """
    if arguments.kernel is None:
        raise ValueError("No kernel argument provided.")

    lengthscale_constraint = _parse_lengthscale_constraint(arguments.lengthscale_bounds)
    kernel_kwargs: dict = {"ard_num_dims": number_train_points}
    if lengthscale_constraint is not None:
        kernel_kwargs["lengthscale_constraint"] = lengthscale_constraint

    kernel = None
    kernel_name =""
    match arguments.kernel:
        case "RBF":
            # to fix output scale, dont wrap in scale kernel, make adj via "trainable_output_scale parameter"
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,
                gpytorch.kernels.RBFKernel(**kernel_kwargs))
            kernel_name = "RBF"

        case "matern32":
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,
                gpytorch.kernels.MaternKernel(nu=1.5, **kernel_kwargs))
            kernel_name = "Matern32"

        case "RBFKeops":
            kernel = signal_variance_kernelWrap(
                arguments.train_signal_variance,
                RBFKEops(**kernel_kwargs)
            )
            kernel_name = "RBFKeops"

        case "matern32Keops":
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,
                MaternKeops(nu=1.5, **kernel_kwargs))
            kernel_name = "Matern32Keops"

    return kernel, kernel_name