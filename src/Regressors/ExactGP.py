import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from Datasets.RegressionDataset import Dataset


class ExactGPModel(gpytorch.models.ExactGP):
    train_data: Dataset
    test_data: Dataset
    trained: bool

    def __init__(
        self, train_data: Dataset, test_data: Dataset, likelihood, kernel=None
    ):
        super(ExactGPModel, self).__init__(
            train_data.features, train_data.targets, likelihood
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.likelihood = likelihood
        self.train_data = train_data
        self.test_data = test_data
        self.trained = False
        if torch.cuda.is_available():
            self.to("cuda")
            self.likelihood = likelihood.cuda()
            self.train_data = train_data.cuda()
            self.test_data = test_data.cuda()

    def forward(self, x):
        mean_x = self.mean_module(x)
        assert isinstance(mean_x, torch.Tensor), "mean must be a tensor"
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_training(self, optimizer, iterations):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

        for i in range(iterations):
            if is_lbfgs:
                loss = optimizer.step(lambda: self._compute_loss(mll))
            else:
                optimizer.zero_grad()
                loss = self._compute_loss(mll)
                loss.backward()
                optimizer.step()

            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    iterations,
                    loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item(),
                )
            )
            torch.cuda.empty_cache()

        self.trained = True

    def _compute_loss(self, mll):
        output = self(self.train_data.features)
        return -mll(output, self.train_data.targets).mean()

    @property
    def posterior(self):
        if not self.trained:
            raise ValueError(
                "The model needs to be trained first. run .run_training(optimizer, iterations)"
            )
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self(self.test_data.features)
        return posterior

    def run_eval(self, show=False):
        self.eval()
        self.likelihood.eval()

        f_preds = self(self.test_data.features)
        y_preds = self.likelihood(f_preds)

        targets = self.test_data.targets.to(y_preds.mean.device)

        with torch.no_grad():
            mu = y_preds.mean
            sigma = y_preds.stddev
            lower_bound = mu - 1.96 * sigma
            upper_bound = mu + 1.96 * sigma
            print(
                "MAE: {}".format(torch.mean(torch.abs(y_preds.mean - targets))),
                "Targets mean: {}".format(torch.mean(targets)),
            )
            print(
                "NLL: {}".format(
                    -y_preds.to_data_independent_dist().log_prob(targets).mean().item()
                )
            )

            is_covered = (targets >= lower_bound) & (targets <= upper_bound)
            picp = is_covered.float().mean()

            print(f"PICP: {picp:.4f}")

            mse = torch.mean((targets - mu) ** 2)
            rmse = torch.sqrt(mse)

            print(f"RMSE: {rmse.item():.4f}")

        if show:
            with torch.no_grad():
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                lower, upper = y_preds.confidence_region()

                test_data_f = self.test_data.features.cpu().numpy()
                train_data_f = self.train_data.features.cpu().numpy()
                train_data_t = self.train_data.targets.cpu().numpy()

                sort_idx = np.argsort(test_data_f, axis=0).flatten()
                test_features_sorted = test_data_f[sort_idx].flatten()
                y_mean_sorted = y_preds.mean.cpu().numpy()[sort_idx]
                lower_sorted = lower.cpu().numpy()[sort_idx]
                upper_sorted = upper.cpu().numpy()[sort_idx]

                ax.plot(train_data_f, train_data_t, "k*")

                ax.plot(test_features_sorted, y_mean_sorted, "b")

                ax.fill_between(
                    test_features_sorted, lower_sorted, upper_sorted, alpha=0.5
                )
                ax.set_ylim([-3, 3])
                ax.legend(["Observed Data", "Mean", "Confidence"])

                plt.show()
