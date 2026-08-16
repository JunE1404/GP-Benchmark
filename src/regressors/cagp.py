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
from collections.abc import Callable
from scaffolds import LogDetails
from helpers import evaluate_regression


def _patch_keops_covar_funcs():
    """Add diag=True support to the KeOps kernel covar funcs.

    The computation-aware ELBO takes the diagonal of the lazy kernel operator
    (prior variance); with the lazy KeOps path active, the covar func would
    return a LazyTensor that to_dense() cannot materialize. diag=True arrives
    via kwargs and is handled elementwise on the plain tensors instead.
    """

    from gpytorch.kernels.keops import matern_kernel, rbf_kernel

    _orig_rbf = rbf_kernel._covar_func

    def _rbf_covar_func(x1, x2, diag=False, **params):
        if diag:
            return (-((x1 - x2) ** 2).sum(-1) / 2).exp().unsqueeze(-1)
        return _orig_rbf(x1, x2, **params)

    _orig_matern = matern_kernel._covar_func

    def _matern_covar_func(x1, x2, nu=2.5, diag=False, **params):
        if diag:
            sq_distance = ((x1 - x2) ** 2).sum(-1)
            distance = (sq_distance + 1e-20).sqrt()
            exp_component = (-math.sqrt(nu * 2) * distance).exp()
            if nu == 0.5:
                constant_component = 1
            elif nu == 1.5:
                constant_component = (math.sqrt(3) * distance) + 1
            elif nu == 2.5:
                constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * sq_distance)
            return (constant_component * exp_component).unsqueeze(-1)
        return _orig_matern(x1, x2, nu=nu, **params)

    rbf_kernel._covar_func = _rbf_covar_func
    matern_kernel._covar_func = _matern_covar_func


_patch_keops_covar_funcs()
#Credit AI Agent, Deepseek V4-Flash

class CAGPModel(ComputationAwareGP):
    train_data: tuple[Tensor, Tensor]
    test_data: tuple[Tensor, Tensor]
    val_data: tuple[Tensor, Tensor]
    trained: bool

    def __init__(
        self,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
        val_data: tuple[Tensor, Tensor],
        projection_dim: int,
        likelihood: None | Likelihood,
        kernel=None,
        mean_module=None,
        device="",
    ):
        """Initialize the Exact GP model.

        Args:
            train_data: Tuple of (train_features, train_targets).
            test_data: Tuple of (test_features, test_targets).
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Optional custom kernel; defaults to ScaleKernel(RBFKernel()).
        """
        super(CAGPModel, self).__init__(
            train_inputs=train_data[0],
            train_targets=train_data[1],
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
            self.train_data = (train_data[0].cuda(), train_data[1].cuda())
            self.test_data = (test_data[0].cuda(), test_data[1].cuda())
            self.val_data = (val_data[0].cuda(), val_data[1].cuda())

    def __str__(self) -> str:
        return "CAGP"

    @contextlib.contextmanager
    def _settings_context(self):
        """Context manager that forces the lazy KeOps kernel path."""
        with gpytorch.settings.max_cholesky_size(0):
            yield

    def forward(self, x):
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

    def run_training(self, optimizer,y_mean, y_std,standardize_val_targets, iterations, logger: Callable[[LogDetails]]):
        """Train the Exact GP model.

        Optimizes kernel hyperparameters and likelihood noise by minimizing
        the negative exact marginal log-likelihood.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            iterations: Number of optimization iterations.
        """

        with self._settings_context():

            def _compute_loss(mll, x_train, y_train):
                """Compute the negative marginal log-likelihood loss."""
                output = self(x_train)
                return -mll(output, y_train).mean()

            self.train()
            self.likelihood.train()
            mll = ComputationAwareELBO(self.likelihood, self)

            is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
            for i in range(iterations):
                if is_lbfgs:

                    def closure():
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


                x = self.val_data[0]
                if next(self.parameters()).is_cuda:
                    x = x.cuda()
                with self._settings_context():
                    self.eval()
                    self.likelihood.eval()
                    with torch.no_grad():
                        posterior = self.likelihood(self(x))

                pst_t = posterior.mean.detach().cpu()
                pred_std = posterior.stddev.detach().cpu()

                MAE, NLL, PICP, RMSE, LScale = evaluate_regression(self,posterior, self.val_data[1], y_mean, y_std, standardize_val_targets)


                logdetails = LogDetails(iteration=i,
                                loss=loss.item(),
                                lengthscale=LScale,
                                likelyhood_noise=self.likelihood.noise.item(),
                                val_MAE=MAE,
                                val_NLL=NLL,
                                val_PICP=PICP,
                                val_RMSE=RMSE,
                                val_pred_mean=pst_t.mean().item(),
                                val_pred_median=pst_t.median().item(),
                                val_pred_q05=torch.quantile(pst_t, 0.05).item(),
                                val_pred_q95=torch.quantile(pst_t, 0.95).item(),
                                val_pred_std_mean=pred_std.mean().item()
                            )
                logger(logdetails)
                self.train()
                self.likelihood.train()
                torch.cuda.empty_cache()

            self.trained = True

    def predict(self, x):
        """Get the posterior distribution over test points after training.

        Returns:
            MultivariateNormal distribution over test targets.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError(
                "The model needs to be trained first. run .run_training(optimizer, iterations)"
            )
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        with self._settings_context():
            self.eval()
            self.likelihood.eval()
            with torch.no_grad():
                posterior = self.likelihood(self(x))
            return posterior
