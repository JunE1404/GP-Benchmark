import gpytorch
import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from .additional_sources.gps.computaion_aware import ComputationAwareGP
from .additional_sources.mlls.comp_aware_elbo import ComputationAwareELBO


class CAGPModel(ComputationAwareGP):
    train_data: tuple[Tensor, Tensor]
    test_data: tuple[Tensor, Tensor]
    trained: bool

    def __init__(
        self,
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
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
        self.trained = False
        if device == "cuda" and torch.cuda.is_available():
            self.to("cuda")
            self.likelihood = likelihood.cuda()
            self.train_data = (train_data[0].cuda(), train_data[1].cuda())
            self.test_data = (test_data[0].cuda(), test_data[1].cuda())

    def __str__(self) -> str:
        return "CAGP"

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
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self(x))
        return posterior
