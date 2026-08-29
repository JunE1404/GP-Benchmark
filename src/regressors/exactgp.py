import gpytorch
import torch
import time
from torch import Tensor
import contextlib
from collections.abc import Callable
from scaffolds import LogDetails
from helpers import evaluate_regression

class ExactGPModel(gpytorch.models.ExactGP):
    train_data: tuple[Tensor, Tensor]
    test_data: tuple[Tensor, Tensor]
    trained: bool

    def __init__(
        self,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
        val_data: tuple[Tensor, Tensor],
        likelihood,
        kernel=None,
        mean_module=None,
        device="",
    ):
        """Initialize the Exact GP model.

        Args:
            train_data: Tuple of (train_features, train_targets).
            test_data: Tuple of (test_features, test_targets).
            val_data: Tuple of (val_features, val_targets).
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Optional custom kernel; defaults to ScaleKernel(RBFKernel()).
        """
        super(ExactGPModel, self).__init__(train_data[0], train_data[1], likelihood)
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

    @contextlib.contextmanager
    def _settings_context(self):
        """Context manager that applies exact-GP settings (Cholesky everywhere)."""
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ), gpytorch.settings.max_cholesky_size(float("inf")):
            yield

    def __str__(self) -> str:
        return "ExactGP"

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

    def run_training(self, optimizer, y_mean, y_std, standardize_val_targets, iterations, logger: Callable[[LogDetails]]):
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
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

            def closure():
                """Closure for LBFGS that zeroes gradients, computes loss, and backpropagates."""
                optimizer.zero_grad()
                loss = _compute_loss(mll, self.train_data[0], self.train_data[1])
                loss.backward()
                return loss

            for i in range(iterations):
                start_time_it = time.perf_counter()
                if is_lbfgs:

                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = _compute_loss(mll, self.train_data[0], self.train_data[1])
                    loss.backward()
                    optimizer.step()
                
                end_step_time = time.perf_counter()

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

                MAE, NLL, PICP, RMSE, LScale = evaluate_regression(self, posterior, self.val_data[1], y_mean, y_std, standardize_val_targets)
                end_iter_time = time.perf_counter()
                logdetails = LogDetails(iteration=i,
                                loss=loss.item(),
                                lengthscale=LScale,
                                likelyhood_noise=self.likelihood.noise.item(),
                                val_MAE=MAE,
                                val_NLL=NLL,
                                val_PICP50=PICP[0.5],
                                val_PICP90=PICP[0.9],
                                val_PICP95=PICP[0.95],
                                val_RMSE=RMSE,
                                it_time_training=end_step_time-start_time_it,
                                it_time=end_iter_time-start_time_it
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
        with self._settings_context():
            if torch.cuda.is_available():
                x = x.cuda()
            self.eval()
            self.likelihood.eval()
            with torch.no_grad():
                posterior = self.likelihood(self(x))
            return posterior
