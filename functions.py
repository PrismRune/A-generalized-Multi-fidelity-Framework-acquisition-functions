import torch
from torch import Tensor
import numpy as np

class Function:
    def __init__(self, dimension: int, name: str, original_bounds: Tensor = None, seed: int = None):
        self.dimension = dimension
        self.name = name
        # Normalized bounds automatically from dimension
        self.normalized_bounds = torch.stack([
            torch.zeros(dimension, dtype=torch.double),
            torch.ones(dimension, dtype=torch.double)
        ])

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
    
    def reset_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def high_fidelity(self, X: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses should implement high-fidelity evaluation.")

    def low_fidelity(self, X: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses should implement low-fidelity evaluation.")

class Forrester(Function):
    def __init__(self):
        super().__init__(dimension=1)

    def high_fidelity(self, X: Tensor) -> Tensor:
        return ((6 * X - 2) ** 2) * torch.sin(12 * X - 4)

    def low_fidelity(self, X: Tensor) -> Tensor:
        return 0.5 * self.high_fidelity(X) + 10 * (X - 0.5)


class Branin(Function):
    def __init__(self):
        super().__init__(dimension=2, name="Branin")
        self.optimum = 0.397887
        self.fidelity_costs = [1, 1, 1]
        self.expected_costs = [1, 5, 10]

        # bounds for denormalization
        self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double)

    def _scale_inputs(self, X: np.ndarray):
        """Scale normalized inputs [0,1] to Branin domain."""
        lb = self.bounds[0].numpy()
        ub = self.bounds[1].numpy()
        X_scaled = X * (ub - lb) + lb
        x1, x2 = X_scaled[:, 0], X_scaled[:, 1]
        return x1, x2

    def high_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        x1, x2 = self._scale_inputs(X_np)

        pi = np.pi
        a = -1.275 / (pi**2)
        b = 5.0 / pi
        c = 6.0
        d = 10.0 - 5.0 / (4.0 * pi)
        e = 10.0

        result = (a * x1**2 + b * x1 + x2 - c)**2 + d * np.cos(x1) + e
        return torch.tensor(result, dtype=torch.double)

    def med_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        x1, x2 = self._scale_inputs(X_np)

        # Use high_fidelity internally, inputs shifted like original
        hf_input = X_np - 2  # keep normalized input shift consistent
        hf_val = self.high_fidelity(torch.tensor(hf_input, dtype=torch.double))
        result = 10 * torch.sqrt(hf_val) + 2 * (x1 - 0.5) - 3 * (3 * x2 - 1) - 1
        return result

    def low_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        x1, x2 = self._scale_inputs(X_np)

        # Use med_fidelity internally
        mf_input = 1.2 * (X_np + 2)
        mf_val = self.med_fidelity(torch.tensor(mf_input, dtype=torch.double))
        result = mf_val - 3 * x2 + 1
        return result
    

class Borehole(Function):
    def __init__(self):
        super().__init__(dimension=8, name="Borehole")
        self.optimum = 3.0957562923431396
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]

    def _scale_inputs(self, X: np.ndarray):
        """Scale normalized inputs to problem domain."""
        x1 = X[:, 0] * 0.1 + 0.05
        x2 = X[:, 1] * (50000 - 100) + 100
        x3 = (X[:, 2] * (115.6 - 63.07) + 63.07) * 1000
        x4 = X[:, 3] * (1110 - 990) + 990
        x5 = X[:, 4] * (116 - 63.1) + 63.1
        x6 = X[:, 5] * (820 - 700) + 700
        x7 = X[:, 6] * (1680 - 1120) + 1120
        x8 = X[:, 7] * (12045 - 9855) + 9855
        return x1, x2, x3, x4, x5, x6, x7, x8

    def high_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        x1, x2, x3, x4, x5, x6, x7, x8 = self._scale_inputs(X_np)

        numerator = 2 * np.pi * x3 * (x4 - x6)
        denominator = np.log(x2 / (x1 + 1e-5)) * (
            1
            + (2 * x7 * x3)
            / (np.log(x2 / (x1 + 1e-5)) * x1**2 * x8 + 1e-5)
            + x3 / (x5 + 1e-5)
        )

        return torch.tensor(numerator / denominator / 100, dtype=torch.double)

    def low_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        x1, x2, x3, x4, x5, x6, x7, x8 = self._scale_inputs(X_np)

        numerator = 5 * x3 * (x4 - x6)
        denominator = np.log(x2 / (x1 + 1e-5)) * (
            1.5
            + (2 * x7 * x3)
            / (np.log(x2 / (x1 + 1e-5)) * x1**2 * x8 + 1e-5)
            + x3 / (x5 + 1e-5)
        )

        return torch.tensor(numerator / denominator / 100, dtype=torch.double)
    

class Hartmann(Function):
    def __init__(self):
        super().__init__(dimension=6, name='Hartmann')
        self.A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        self.P = (1e-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.delta = np.array([0.01, -0.01, -0.1, 0.1])

    def high_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        sum1 = 0
        for i in range(4):
            sum2 = 0
            for j in range(self.dimension):
                sum2 += self.A[i, j] * (X_np[:, j] - self.P[i, j])**2
            sum1 += self.alpha[i] * np.exp(-sum2)
        return torch.tensor(sum1, dtype=torch.double)

    def low_fidelity(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        sum1 = 0
        for i in range(4):
            sum2 = 0
            for j in range(self.dimension):
                sum2 += self.A[i, j] * (X_np[:, j] - self.P[i, j])**2
            sum1 += (self.alpha[i] + 2 * self.delta[i]) * np.exp(-sum2)
        return torch.tensor(sum1, dtype=torch.double)