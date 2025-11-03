import torch
import gpytorch

class RelationshipModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.train_X = None
        self.train_Y = None
        self.trained = False

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None

        self.a = None

    def _initialize_model(self):
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.model = ExactGPModel(self.train_X, self.train_Y, self.likelihood)

    def add_point(self, x, y_low, y_high):
        # Convert inputs to tensors
        x = torch.tensor(x, dtype=torch.float64).flatten()
        y_low = torch.tensor([y_low], dtype=torch.float64)
        y_high = torch.tensor([y_high], dtype=torch.float64)

        # Stack x and y_low as model input
        new_input = torch.cat([x, y_low]).unsqueeze(0)  # shape (1, input_dim+1)

        if self.train_X is None:
            self.train_X = new_input
            self.train_Y = y_high
        else:
            self.train_X = torch.cat([self.train_X, new_input], dim=0)
            self.train_Y = torch.cat([self.train_Y, y_high], dim=0)

        # Initialize or update model
        self._initialize_model()

        # Train the GP
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 50
        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -mll(output, self.train_Y)
            loss.backward()
            optimizer.step()

        self.trained = True

    def estimate_hf(self, x, y_low, return_std=False):
        if not self.trained:
            raise RuntimeError("Model is not trained.")

        x = torch.tensor(x, dtype=torch.float64).flatten()
        y_low = torch.tensor([y_low], dtype=torch.float64)
        x_test = torch.cat([x, y_low]).unsqueeze(0)

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x_test))
            mean = posterior.mean.item()
            std = posterior.variance.sqrt().item()

        return (mean, std) if return_std else mean