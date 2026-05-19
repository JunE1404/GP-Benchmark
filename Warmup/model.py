import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_data, test_data, likelihood):
        super(ExactGPModel, self).__init__(
            train_data.features, train_data.targets, likelihood
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood
        self.train_data = train_data
        self.test_data = test_data
        if torch.cuda.is_available():
            self = self.cuda()
            self.likelihood = likelihood.cuda()

    def forward(self, x):
        mean_x = self.mean_module(x)
        assert isinstance(mean_x, torch.Tensor), "mean must be a tensor"
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_training(self, optimizer, iterations):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        training_iter = iterations

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self(self.train_data.features)
            loss = -mll(output, self.train_data.targets)

            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item(),
                )
            )
            optimizer.step()
            torch.cuda.empty_cache()

    def run_eval(self, show=False):
        self.eval()
        self.likelihood.eval()

        f_preds = self(self.test_data.features)
        y_preds = self.likelihood(f_preds)

        # f_mean = f_preds.mean
        # f_var = f_preds.variance
        # f_covar = f_preds.covariance_matrix
        # f_samples = f_preds.sample(sample_shape=torch.Size((1000,)))
        with torch.no_grad():
            mu = y_preds.mean.cpu()
            sigma = y_preds.stddev.cpu()
            lower_bound = mu - 1.96 * sigma
            upper_bound = mu + 1.96 * sigma
            print(
                "MAE: {}".format(
                    torch.mean(torch.abs(y_preds.mean - self.test_data.targets))
                ),
                "Targets mean: {}".format(torch.mean(self.test_data.targets)),
            )
            print(
                "NLL: {}".format(
                    -y_preds.to_data_independent_dist()
                    .log_prob(self.test_data.targets)
                    .mean()
                    .item()
                )
            )

            is_covered = (self.test_data.targets.cpu() >= lower_bound) & (
                self.test_data.targets.cpu() <= upper_bound
            )
            picp = is_covered.float().mean()

            print(f"PICP: {picp:.4f}")

            mse = torch.mean((self.test_data.targets.cpu() - mu) ** 2)
            rmse = torch.sqrt(mse)

            print(f"RMSE: {rmse.item():.4f}")

        if show:
            with torch.no_grad():
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                lower, upper = y_preds.confidence_region()

                test_data_f = self.test_data.features.cpu().numpy()
                test_data_t = self.test_data.targets.cpu().numpy()
                train_data_f = self.train_data.features.cpu().numpy()
                train_data_t = self.train_data.targets.cpu().numpy()

                sort_idx = np.argsort(test_data_f, axis=0).flatten()
                test_features_sorted = test_data_f[sort_idx]
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
