import gpytorch
import torch
import time
from torch import Tensor
import contextlib
from collections.abc import Callable, Iterator
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
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        device: str = "",
    ) -> None:
        """Initialize the exact GP model.

        Args:
            train_data: Tuple of (train_features, train_targets).
            test_data: Tuple of (test_features, test_targets).
            val_data: Tuple of (val_features, val_targets).
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Covariance kernel; must be provided.
            mean_module: Mean module; must be provided.
            device: ``"cuda"`` moves the model and data to GPU when available.

        Raises:
            ValueError: If ``mean_module``, ``kernel`` or ``likelihood`` is
                ``None``.
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
    def _settings_context(self) -> Iterator[None]:
        """Context manager that applies exact-GP settings (Cholesky everywhere)."""
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ), gpytorch.settings.max_cholesky_size(float("inf")):
            yield

    def __str__(self) -> str:
        """Return the display name used in result file paths."""
        return "ExactGP"

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

    def run_training(self, optimizer: torch.optim.Optimizer, y_mean: Tensor, y_std: Tensor, standardize_test_targets: bool, iterations: int, logger: Callable[[LogDetails], None]) -> float:
        """Train the exact GP by maximizing the exact marginal log-likelihood.

        Each iteration takes a gradient step on the training split and logs
        test-split metrics. The returned duration is ``start - end`` (negative),
        matching the convention used across the benchmark.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            y_mean: Training target mean used to un-standardize metrics.
            y_std: Training target standard deviation used to un-standardize metrics.
            standardize_test_targets: Whether the test targets are standardized.
            iterations: Number of optimization iterations.
            logger: Callback invoked with a :class:`LogDetails` after each iteration.

        Returns:
            Negative wall-clock training duration in seconds.
        """
        time_start = time.time()
        with self._settings_context():
            def _compute_loss(mll: gpytorch.mlls.ExactMarginalLogLikelihood, x_train: Tensor, y_train: Tensor) -> Tensor:
                """Compute the negative marginal log-likelihood loss."""
                output = self(x_train)
                return -mll(output, y_train).mean()

            self.train()
            self.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

            def closure() -> Tensor:
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

                x = self.test_data[0]
                if next(self.parameters()).is_cuda:
                    x = x.cuda()
                with self._settings_context():
                    self.eval()
                    self.likelihood.eval()
                    with torch.no_grad():
                        posterior = self.likelihood(self(x))

                pst_t = posterior.mean.detach().cpu()
                pred_std = posterior.stddev.detach().cpu()

                MAE, NLL, PICP, RMSE, LScale = evaluate_regression(self, posterior, self.test_data[1], y_mean, y_std, standardize_test_targets)
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
            return time_start-time_end

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
        time_start = time.time()
        with self._settings_context():
            if torch.cuda.is_available():
                x = x.cuda()
            self.eval()
            self.likelihood.eval()
            with torch.no_grad():
                posterior = self.likelihood(self(x))
            time_end = time.time()
            return posterior, time_start-time_end
