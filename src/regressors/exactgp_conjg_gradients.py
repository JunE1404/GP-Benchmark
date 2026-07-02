import math

import gpytorch
import torch
from linear_operator.utils import StochasticLQ
from linear_operator.utils.linear_cg import linear_cg
from torch import Tensor


class ExactGPConjGradients(gpytorch.models.ExactGP):
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
        super(ExactGPConjGradients, self).__init__(
            train_data[0], train_data[1], likelihood
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

    def run_training(self, optimizer, iterations):
        """Train the Exact GP model.

        Optimizes kernel hyperparameters and likelihood noise by minimizing
        the negative exact marginal log-likelihood.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            iterations: Number of optimization iterations.
        """

        def _compute_loss_cg(x_train, y_train):
            N = x_train.shape[0]
            noise = self.likelihood.noise

            def matmul(v):
                return self.covar_module(x_train, x_train) @ v + noise * v

            y_centered = y_train - self.mean_module(x_train)
            # CG solve
            alpha = linear_cg(matmul, y_centered, max_iter=50)

            # Log-det
            slq = StochasticLQ(max_iter=20, num_random_probes=10)
            logdet = slq.evaluate(
                matmul, matrix_shape=(N, N), funcs=[lambda x: x.log()]
            )

            # Cache α for prediction
            self._alpha = alpha

            return 0.5 * (y_centered @ alpha + logdet + N * math.log(2 * math.pi))

        self.train()
        self.likelihood.train()

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

        for i in range(iterations):
            if is_lbfgs:

                def closure():
                    """Closure for LBFGS that zeroes gradients, computes loss, and backpropagates."""
                    optimizer.zero_grad()
                    loss = _compute_loss_cg(self.train_data[0], self.train_data[1])
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = _compute_loss_cg(self.train_data[0], self.train_data[1])
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
        if torch.cuda.is_available():
            x = x.cuda()
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            prior_mean = self.mean_module(x)
            k_x_train = self.covar_module(x, self.train_data[0])
            pred_mean = prior_mean + k_x_train @ self._alpha
            pred_covar = self.covar_module(x)
            # no predictive varaince
        return gpytorch.distributions.MultivariateNormal(pred_mean, pred_covar)
