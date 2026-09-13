from typing_extensions import Tuple
from gpytorch.kernels.keops import MaternKernel as MaternKeops
from gpytorch.kernels.keops import RBFKernel as RBFKEops
import gpytorch

from scaffolds import RunArguments

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


def getKernel(
    arguments: RunArguments, number_train_points: int
) -> Tuple[gpytorch.kernels.Kernel | None, str]:
    """Build the covariance kernel named by ``arguments.kernel``.

    Args:
        arguments: Run configuration. ``arguments.kernel`` selects the kernel.
        number_train_points: Number of input dimensions, used for ARD kernels.

    Returns:
        Tuple of ``(kernel, display_name)``. The kernel is ``None`` when the name
        matches no supported kernel.

    Raises:
        ValueError: If ``arguments.kernel`` is ``None``.
    """
    if arguments.kernel is None:
        raise ValueError("No kernel argument provided.")
    kernel = None
    kernel_name =""
    match arguments.kernel:
        case "RBF":
            # to fix output scale, dont wrap in scale kernel, make adj via "trainable_output_scale parameter"
            kernel = signal_variance_kernelWrap(gpytorch.kernels.RBFKernel(
                    ard_num_dims=number_train_points,
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(10e-6),
                ))
            kernel_name = "RBF"

        case "matern2.5":
            kernel = signal_variance_kernelWrap(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
            kernel_name = "Matern 2.5"

        case "RBFKeops":
            kernel = signal_variance_kernelWrap(
                RBFKEops(
                    ard_num_dims=number_train_points,
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(10e-6),
                )
            )
            kernel_name = "RBFKeops"

        case "matern2.5Keops":
            kernel = signal_variance_kernelWrap(MaternKeops(nu=2.5))
            kernel_name = "Matern 2.5 Keops"

    return kernel, kernel_name