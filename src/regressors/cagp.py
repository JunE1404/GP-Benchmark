from misc.scaffolds import DatasetData
from misc.scaffolds import Split
from regressors.regressor import Regressor
import time
import contextlib
import math
from typing import List, Tuple

import gpytorch
import torch
import wandb
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from gpytorch.models import ComputationAwareGP
from gpytorch.mlls import ComputationAwareELBO
from collections.abc import Callable, Iterator
from misc.scaffolds import LogDetails
from eval.evaluate_regression import evaluate_regression


def _patch_keops_covar_funcs() -> None:
    """Add diag=True support to the KeOps kernels (covar funcs and forward).

    The computation-aware ELBO takes the diagonal of the lazy kernel operator
    (prior variance); with the lazy KeOps path active, the covar func would
    return a LazyTensor that to_dense() cannot materialize. Without a
    ScaleKernel wrapper, the diagonal request additionally reaches the kernel
    forward directly (LazyEvaluatedKernelTensor), which must return the 1-D
    diagonal. Both cases are handled elementwise on the plain tensors.
    """

    from gpytorch.kernels.keops import matern_kernel, rbf_kernel

    def _rbf_diag(x1: Tensor, x2: Tensor) -> Tensor:
        """Return the elementwise RBF covariance diagonal between ``x1`` and ``x2``."""
        return (-((x1 - x2) ** 2).sum(-1) / 2).exp()

    def _matern_diag(x1: Tensor, x2: Tensor, nu: float) -> Tensor:
        """Return the elementwise Matern covariance diagonal for smoothness ``nu``."""
        sq_distance = ((x1 - x2) ** 2).sum(-1)
        distance = (sq_distance + 1e-20).sqrt()
        exp_component = (-math.sqrt(nu * 2) * distance).exp()
        if nu == 0.5:
            constant_component = 1
        elif nu == 1.5:
            constant_component = (math.sqrt(3) * distance) + 1
        elif nu == 2.5:
            constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * sq_distance)
        return constant_component * exp_component

    _orig_rbf = rbf_kernel._covar_func

    def _rbf_covar_func(x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """KeOps RBF covar func with a working ``diag=True`` branch."""
        if diag:
            return _rbf_diag(x1, x2).unsqueeze(-1)
        return _orig_rbf(x1, x2, **params)

    _orig_matern = matern_kernel._covar_func

    def _matern_covar_func(x1: Tensor, x2: Tensor, nu: float = 2.5, diag: bool = False, **params) -> Tensor:
        """KeOps Matern covar func with a working ``diag=True`` branch."""
        if diag:
            return _matern_diag(x1, x2, nu).unsqueeze(-1)
        return _orig_matern(x1, x2, nu=nu, **params)

    _orig_rbf_forward = rbf_kernel.RBFKernel.forward

    def _rbf_forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> Tensor:
        """RBF kernel forward that returns the 1-D diagonal when ``diag=True``."""
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        if diag:
            return _rbf_diag(x1_, x2_)
        return _orig_rbf_forward(self, x1, x2, **kwargs)

    _orig_matern_forward = matern_kernel.MaternKernel.forward

    def _matern_forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> Tensor:
        """Matern kernel forward that returns the 1-D diagonal when ``diag=True``."""
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean) / self.lengthscale
        x2_ = (x2 - mean) / self.lengthscale
        if diag:
            return _matern_diag(x1_, x2_, self.nu)
        return _orig_matern_forward(self, x1, x2, **kwargs)

    rbf_kernel._covar_func = _rbf_covar_func
    matern_kernel._covar_func = _matern_covar_func
    rbf_kernel.RBFKernel.forward = _rbf_forward
    matern_kernel.MaternKernel.forward = _matern_forward


_patch_keops_covar_funcs()
#Credit AI Agent, Deepseek V4-Flash

class CAGPModel(ComputationAwareGP, Regressor):
    datasetData: DatasetData

    def __init__(
        self,
        dataset: DatasetData,
        projection_dim: int,
        likelihood: None | Likelihood,
        kernel: gpytorch.kernels.Kernel | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        device: str = "",
    ) -> None:
        """Initialize the computation-aware GP (CAGP) model.

        Args:
            dataset: Dataset splits plus train target statistics and
                standardization flags.
            projection_dim: Projection dimension / number of actions used by the
                computation-aware ELBO.
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Covariance kernel; must be provided.
            mean_module: Mean module; must be provided.
            device: ``"cuda"`` moves the model and data to GPU when available.

        Raises:
            ValueError: If ``mean_module``, ``kernel`` or ``likelihood`` is
                ``None``.
        """
        self.datasetData = dataset
        train_data  = self.datasetData.train_data
        val_data = self.datasetData.val_data
        test_data = self.datasetData.test_data
        super(CAGPModel, self).__init__(
            train_inputs=dataset.train_data.features,
            train_targets=dataset.train_data.targets,
            mean_module=mean_module,
            covar_module=kernel,
            likelihood=likelihood,
            projection_dim=projection_dim,
        )
        if mean_module is None:
            raise ValueError("No mean module set.")
        else:
            self.mean_module = mean_module
        if kernel is None:
            raise ValueError("No kernel (covar module) set.")
        else:
            self.covar_module = kernel
        if likelihood is None:
            raise ValueError("No likelyhood set.")
        else:
            self.likelihood = likelihood

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.trained = False
        if device == "cuda" and torch.cuda.is_available():
            self.to("cuda")
            self.likelihood = likelihood.cuda()
            self.datasetData = self.datasetData.to("cuda")
            self.train_data = (self.datasetData.train_data.features, self.datasetData.train_data.targets)
            self.test_data = (self.datasetData.test_data.features, self.datasetData.test_data.targets)
            self.val_data = (self.datasetData.val_data.features, self.datasetData.val_data.targets)

    def __str__(self) -> str:
        """Return the display name used in result file paths."""
        return "CAGP"

    @contextlib.contextmanager
    def _settings_context(self) -> Iterator[None]:
        """Context manager that forces the lazy KeOps kernel path."""
        with gpytorch.settings.max_cholesky_size(0):
            yield

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior/posterior GP distribution at input points.

        Args:
            x: Input tensor of shape (n_samples, n_features).

        Returns:
            MultivariateNormal distribution with the GP mean and covariance.
        """
        mean_x = self.mean_module(x)
        assert isinstance(mean_x, torch.Tensor), "mean must be a tensor"
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_training(self, optimizer: torch.optim.Optimizer,iterations: int, logger: Callable[[LogDetails], None]) -> float:
        """Train the CAGP model with the computation-aware ELBO.

        Each iteration takes a gradient step on the training split and logs
        test-split metrics. The returned duration is ``start - end`` (negative),
        matching the convention used across the benchmark.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            iterations: Number of optimization iterations.
            logger: Callback invoked with a :class:`LogDetails` after each iteration.

        Returns:
            Negative wall-clock training duration in seconds.
        """
        time_start = time.time()
        with self._settings_context():

            def _compute_loss(mll: ComputationAwareELBO, x_train: Tensor, y_train: Tensor) -> Tensor:
                """Compute the negative marginal log-likelihood loss."""
                output = self(x_train)
                return -mll(output, y_train).mean()

            self.train()
            self.likelihood.train()
            mll = ComputationAwareELBO(self.likelihood, self)

            is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
            for i in range(iterations):
                start_time_it = time.perf_counter()
                if is_lbfgs:

                    def closure() -> Tensor:
                        """Closure for LBFGS that zeroes gradients, computes loss, and backpropagates."""
                        optimizer.zero_grad()
                        loss = _compute_loss(mll, self.train_data[0], self.train_data[1])
                        loss.backward()
                        return loss

                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = _compute_loss(mll, self.train_data[0], self.train_data[1])
                    loss.backward()
                    optimizer.step()


                end_step_time = time.perf_counter()
                x = self.test_data[0]
                if next(self.parameters()).is_cuda:
                    x = x.cuda()
                with self._settings_context():
                    self.eval()
                    self.likelihood.eval()
                    with torch.no_grad():
                        posterior = self.likelihood(self(x))
                        
                MAE, NLL, PICP, RMSE, LScale = evaluate_regression(self, posterior, datasetData=self.datasetData)

                end_iter_time = time.perf_counter()
                if hasattr(self.covar_module, "outputscale"):
                    outputscale = self.covar_module.outputscale.item()
                else:
                    outputscale = 1
                logdetails = LogDetails(iteration=i,
                                loss=loss.item(),
                                lengthscale=LScale,
                                outputscale=outputscale,
                                likelyhood_noise=self.likelihood.noise.item(),
                                test_MAE=MAE,
                                test_NLL=NLL,
                                test_PICP50=PICP[0.5],
                                test_PICP90=PICP[0.9],
                                test_PICP95=PICP[0.95],
                                test_RMSE=RMSE,
                                it_time_training=end_step_time-start_time_it,
                                it_time=end_iter_time-start_time_it
                            )
                logger(logdetails)
                self.train()
                self.likelihood.train()
                torch.cuda.empty_cache()

            self.trained = True
            time_end = time.time()
            return time_end -time_start

    def predict(self, x: Tensor) -> tuple[gpytorch.distributions.MultivariateNormal, float]:
        """Get the posterior distribution over test points after training.

        Args:
            x: Input tensor of shape (n_samples, n_features).

        Returns:
            Tuple of the posterior distribution and the negative wall-clock
            prediction duration in seconds.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        
        if not self.trained:
            raise ValueError(
                "The model needs to be trained first. run .run_training(optimizer, iterations)"
            )
            
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        time_start = time.time()
        with self._settings_context():
            self.eval()
            self.likelihood.eval()
            with torch.no_grad():
                posterior = self.likelihood(self(x))
            time_end = time.time()
            return posterior, time_start-time_end
