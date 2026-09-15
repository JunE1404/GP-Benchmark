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
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,gpytorch.kernels.RBFKernel(
                    ard_num_dims=number_train_points,
                ))
            kernel_name = "RBF"

        case "matern32":
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,
                gpytorch.kernels.MaternKernel(
                    nu=1.5,
                    ard_num_dims=number_train_points,
                )
            )
            kernel_name = "Matern 2.5"

        case "RBFKeops":
            kernel = signal_variance_kernelWrap(
                arguments.train_signal_variance,
                RBFKEops(
                    ard_num_dims=number_train_points,
                )
            )
            kernel_name = "RBFKeops"

        case "matern32Keops":
            kernel = signal_variance_kernelWrap(arguments.train_signal_variance,
                MaternKeops(
                    nu=1.5,
                    ard_num_dims=number_train_points,
                )
            )
            kernel_name = "Matern 1.5 Keops"

    return kernel, kernel_name