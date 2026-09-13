import gpytorch
import torch
import time
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from collections.abc import Callable
from misc.scaffolds import LogDetails

from misc.helpers import getInducingPoints
from misc.evaluate_regression import evaluate_regression


class SparseVariationalGP(ApproximateGP):
    def __init__(
        self,
        strategy: str,
        strategy_seed: int,
        n: int,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
        val_data: tuple[Tensor, Tensor],
        batch_size: int,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        device: str = "",
    ) -> None:
        """Initialize the sparse variational GP (SVGP) model.

        Args:
            strategy: Inducing-point initialization method, ``"kmeans"`` or
                ``"random"``.
            strategy_seed: Seed used by the inducing-point initialization.
            n: Number of inducing points.
            train_data: Tuple of (train_features, train_targets).
            test_data: Tuple of (test_features, test_targets).
            val_data: Tuple of (val_features, val_targets).
            batch_size: Minibatch size used during training.
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Covariance kernel; must be provided.
            mean_module: Mean module; must be provided.
            device: ``"cuda"`` moves the model and data to GPU when available.

        Raises:
            ValueError: If ``mean_module``, ``kernel`` or ``likelihood`` is
                ``None``.
        """

        inducing_points = getInducingPoints(train_data[0], n, strategy=strategy, seed=strategy_seed)
        self.inducing_point_strat = strategy

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(SparseVariationalGP, self).__init__(variational_strategy)
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
        self.batch_size = batch_size
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

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior GP distribution at input points.

        Args:
            x: Input tensor of shape (n_samples, n_features).

        Returns:
            MultivariateNormal distribution with the GP mean and covariance.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_training(self, optimizer: torch.optim.Optimizer, y_mean: Tensor, y_std: Tensor, standardize_test_targets: bool, iterations: int, logger: Callable[[LogDetails], None]) -> float:
        """Train the SVGP model using minibatch variational inference.

        Optimizes kernel hyperparameters, likelihood noise, inducing point
        locations, and variational parameters by minimizing the negative
        variational ELBO. The returned duration is ``start - end`` (negative),
        matching the convention used across the benchmark.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            y_mean: Training target mean used to un-standardize metrics.
            y_std: Training target standard deviation used to un-standardize metrics.
            standardize_test_targets: Whether the test targets are standardized.
            iterations: Number of passes (epochs) through the training data.
            logger: Callback invoked with a :class:`LogDetails` after each epoch.

        Returns:
            Negative wall-clock training duration in seconds.
        """

        time_start = time.time()

        def _compute_loss(mll: gpytorch.mlls.VariationalELBO, x_batch: Tensor, y_batch: Tensor) -> Tensor:
            """Compute the negative variational ELBO loss for a batch."""
            output = self(x_batch)
            return -mll(output, y_batch)

        self.train()
        self.likelihood.train()

        train_dataset = TensorDataset(self.train_data[0], self.train_data[1])

        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self, num_data=self.train_data[1].size(0)
        )

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        with gpytorch.settings._linalg_dtype_cholesky(torch.float32):
            for i in range(iterations):
                epoch_loss = 0
                train_loader = DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                ) 
                start_time_it = time.perf_counter()
                for x_batch, y_batch in train_loader:
                    if is_lbfgs:

                        def closure() -> Tensor:
                            """Closure for LBFGS that zeroes gradients, computes loss, and backpropagates."""
                            optimizer.zero_grad()
                            loss = _compute_loss(mll, x_batch, y_batch)
                            loss.backward()
                            return loss

                        loss = optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        loss = _compute_loss(mll, x_batch, y_batch)
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item()


                
                end_step_time = time.perf_counter()

                x = self.test_data[0]
                if next(self.parameters()).is_cuda:
                    x = x.cuda()
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
                                        loss=epoch_loss/len(train_loader),
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
        if torch.cuda.is_available():
            x = x.cuda()
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self(x))
        time_end = time.time()
        return posterior, time_start-time_end

    def __str__(self) -> str:
        """Return the display name used in result file paths."""
        return "SVGP"
