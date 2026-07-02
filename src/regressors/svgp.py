import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class SparseVariationalGP(ApproximateGP):
    def __init__(
        self,
        inducing_points,  # instead pass string for strategy (rand, k-means) + number of points n
        train_data: tuple[Tensor, Tensor],
        test_data: tuple[Tensor, Tensor],
        likelihood,
        kernel=None,
        mean_module=None,
        device="",
    ):
        """Initialize the Sparse Variational GP model.

        Args:
            inducing_points: Initial inducing point locations, shape (m, n_features).
            train_data: Tuple of (train_features, train_targets).
            test_data: Tuple of (test_features, test_targets).
            likelihood: A GPyTorch likelihood (e.g. GaussianLikelihood).
            kernel: Optional custom kernel; defaults to ScaleKernel(RBFKernel()).
        """
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

        self.train_data = train_data
        self.test_data = test_data
        self.trained = False
        if device == "cuda" and torch.cuda.is_available():
            self.to("cuda")
            self.likelihood = likelihood.cuda()
            self.train_data = (train_data[0].cuda(), train_data[1].cuda())
            self.test_data = (test_data[0].cuda(), test_data[1].cuda())

    def forward(self, x):
        """Compute the prior GP distribution at input points.

        Args:
            x: Input tensor of shape (n_samples, n_features).

        Returns:
            MultivariateNormal distribution with the GP mean and covariance.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_training(self, optimizer, iterations):
        """Train the SVGP model using minibatch variational inference.

        Optimizes kernel hyperparameters, likelihood noise, inducing point
        locations, and variational parameters by minimizing the negative
        variational ELBO.

        Args:
            optimizer: A PyTorch optimizer (e.g. Adam or LBFGS).
            epochs: Number of passes through the full training dataset.
        """

        def _compute_loss(mll, x_batch, y_batch):
            """Compute the negative variational ELBO loss for a batch."""
            output = self(x_batch)
            return -mll(output, y_batch).mean()

        self.train()
        self.likelihood.train()

        train_dataset = TensorDataset(self.train_data[0], self.train_data[1])
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )  # batch size chosen with flag

        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self, num_data=self.train_data[1].size(0)
        )

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

        for i in range(iterations):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                if is_lbfgs:

                    def closure():
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

                torch.cuda.empty_cache()
            # print(
            #    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
            #    % (
            #        i + 1,
            #        epochs,
            #        epoch_loss / len(train_loader),
            #        self.covar_module.base_kernel.lengthscale.item(),
            #        self.likelihood.noise.item(),
            #    )
            # )

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
            posterior = self.likelihood(self(x))
        return posterior

    def __str__(self) -> str:
        return "SVGP"
