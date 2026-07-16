import gpytorch
import torch
from torch import Tensor
import contextlib


class ExactGPCGModel(gpytorch.models.ExactGP):
    train_data: tuple[Tensor, Tensor]
    test_data: tuple[Tensor, Tensor]
    trained: bool

    def __init__(
        self,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
        likelihood,
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
        super(ExactGPCGModel, self).__init__(train_data[0], train_data[1], likelihood)
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
        self.trained = False
        if device == "cuda" and torch.cuda.is_available():
            self.to("cuda")
            self.likelihood = likelihood.cuda()
            self.train_data = (train_data[0].cuda(), train_data[1].cuda())
            self.test_data = (test_data[0].cuda(), test_data[1].cuda())

    @contextlib.contextmanager
    def _settings_context(self):
        """Context manager that applies CG-for-solves settings."""
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=True,
            solves=True,
        ), gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.cg_tolerance(1.0), \
            gpytorch.settings.eval_cg_tolerance(1e-12), \
            gpytorch.settings.max_cg_iterations(1000):
            yield

    def __str__(self) -> str:
        return "ExactGPConjGradients"

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

    def run_training(self, optimizer, iterations):
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
                if is_lbfgs:

                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = _compute_loss(mll, self.train_data[0], self.train_data[1])
                    loss.backward()
                    optimizer.step()

                # print(
                #    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                #    % (
                #        i + 1,
                #        iterations,
                #        loss.item(),
                #        self.covar_module.base_kernel.lengthscale.item(),
                #        self.likelihood.noise.item(),
                #    )
                # )
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
