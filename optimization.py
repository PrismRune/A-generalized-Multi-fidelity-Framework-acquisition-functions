from relationship_model import *

import torch
from torch import Tensor
from typing import Callable
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import PosteriorMean
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement, qKnowledgeGradient, qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective

import torch
import numpy as np
import pickle
from functions import *

import warnings
warnings.filterwarnings("ignore")


def run_optimization_high_fidelity(
    seed: int,
    name: str,
    high_budget: int,
    bounds: Tensor,
    high_fidelity_function: Callable[[Tensor], Tensor],
    initial_samples: int,
    dimensions: int,
    acquisition_function: str,
    maximize: bool = False,
) -> pd.DataFrame:

    torch.manual_seed(seed)

    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(initial_samples, dimensions, dtype=torch.float64)
    train_Y_HF = []

    for x in train_X:
        y_hf = high_fidelity_function(x.unsqueeze(0))
        train_Y_HF.append(y_hf.squeeze())

    train_Y_HF = torch.tensor(train_Y_HF).unsqueeze(-1)
    train_X_HF = train_X.clone()

    rows = []

    neg_objective = GenericMCObjective(lambda samples, X=None: -samples[..., 0])
    pos_objective = GenericMCObjective(lambda samples, X=None: samples[..., 0])

    objective = pos_objective if maximize else neg_objective

    for high_count in range(high_budget):
        hf_model = SingleTaskGP(train_X_HF, train_Y_HF)
        mll = ExactMarginalLogLikelihood(hf_model.likelihood, hf_model)
        fit_gpytorch_model(mll)

        mean_acqf = PosteriorMean(hf_model, maximize=maximize)
        best_x, _ = optimize_acqf(
            acq_function=mean_acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            return_best_only=True,
        )

        posterior = hf_model.posterior(best_x)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()
        real_y = high_fidelity_function(best_x).item()

        rows.append({
            'seed': seed,
            'type': 'high',
            'function_name': name, 
            'acquisition_function': acquisition_function,
            'high_experiment': 'n/a',
            'predicted_mean': mean,
            'predicted_std': std,
            'real_y': real_y,
            'num_high_evals': high_count,
            'num_low_evals': 0,
        })

        if acquisition_function == "EI":
            best_f=train_Y_HF.max().item() if maximize else train_Y_HF.min().item()
            acqf = ExpectedImprovement(hf_model, best_f=best_f, maximize=maximize)
        elif acquisition_function == "PI":
            best_f=train_Y_HF.max().item() if maximize else train_Y_HF.min().item()
            acqf = ProbabilityOfImprovement(hf_model, best_f=best_f, maximize=maximize)
        elif acquisition_function == "UCB":
            acqf = UpperConfidenceBound(hf_model, beta=0.2, maximize=maximize)
        elif acquisition_function == "KG":
            acqf = qKnowledgeGradient(hf_model, num_fantasies=64, objective=objective)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        new_y = high_fidelity_function(candidate).view(-1, 1)

        train_X_HF = torch.cat([train_X_HF, candidate], dim=0)
        train_Y_HF = torch.cat([train_Y_HF, new_y], dim=0)

    return pd.DataFrame(rows)


def run_optimization_multi_fidelity(
    seed: int,
    name: str,
    high_budget: int,
    bounds: Tensor,
    high_fidelity_function: Callable[[Tensor], Tensor],
    low_fidelity_function: Callable[[Tensor], Tensor],
    initial_samples: int,
    dimensions: int,
    low_per_high: int,
    acquisition_function: str,
    high_experiment: str,
    maximize: bool = False,
) -> pd.DataFrame:
    
    torch.manual_seed(seed)
    high_low = RelationshipModel(input_dim=dimensions)

    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(initial_samples, dimensions, dtype=torch.float64)

    train_Y_HF = []
    train_Y_LF = []

    for x in train_X:
        y_hf = high_fidelity_function(x.unsqueeze(0))
        y_lf = low_fidelity_function(x.unsqueeze(0))

        train_Y_HF.append(y_hf.squeeze())
        train_Y_LF.append(y_lf.squeeze())
              # shape (N, D)
    train_Y_HF = torch.tensor(train_Y_HF).unsqueeze(-1)
    train_Y_LF = torch.tensor(train_Y_LF).unsqueeze(-1)

    train_X_HF = train_X.clone()
    train_X_LF = train_X.clone()
    train_Y = train_Y_HF.clone()

    for i in range(train_X.shape[0]):
        x = train_X[i]
        y_low = train_Y_LF[i].item()
        y_high = train_Y_HF[i].item()
        high_low.add_point(x, y_low, y_high)

    rows = []

    neg_objective = GenericMCObjective(lambda samples, X=None: -samples[..., 0])
    pos_objective = GenericMCObjective(lambda samples, X=None: samples[..., 0])

    objective = pos_objective if maximize else neg_objective

    for high_count in range(high_budget):

        hf_model = SingleTaskGP(train_X_HF, train_Y_HF)
        hf_mll = ExactMarginalLogLikelihood(hf_model.likelihood, hf_model)
        fit_gpytorch_model(hf_mll)

        mf_model = SingleTaskGP(train_X, train_Y)
        mf_mll = ExactMarginalLogLikelihood(mf_model.likelihood, mf_model)
        fit_gpytorch_model(mf_mll)

        mean_acqf = PosteriorMean(hf_model, maximize=maximize)
        best_x, _ = optimize_acqf(
            acq_function=mean_acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            return_best_only=True,
            maximize = maximize
        )

        posterior = hf_model.posterior(best_x)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()
        real_y = high_fidelity_function(best_x).item()

        rows.append({
            'seed': seed,
            'type': 'combined',
            'function_name': name, 
            'acquisition_function': acquisition_function,
            'high_experiment': high_experiment,
            'predicted_mean': mean,
            'predicted_std': std,
            'real_y': real_y,
            'num_high_evals': high_count,
            'num_low_evals': high_count * low_per_high,
        })

        if acquisition_function == "EI":
            best_f=train_Y_HF.max().item() if maximize else train_Y_HF.min().item()
            acqf = qExpectedImprovement(model=mf_model, best_f=best_f, objective=objective)
        elif acquisition_function == "PI":
            best_f=train_Y_HF.max().item() if maximize else train_Y_HF.min().item()
            acqf = qProbabilityOfImprovement(model=mf_model, best_f=best_f, objective=objective)
        elif acquisition_function == "UCB":
            acqf = qUpperConfidenceBound(mf_model, beta=0.2, objective=objective)
        elif acquisition_function == "KG":
            acqf = qKnowledgeGradient(mf_model, num_fantasies=64, objective=objective)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=low_per_high,
            num_restarts=20,
            raw_samples=1024,
        )

        # Pick candidate with highest HF EI to evaluate high fidelity
        if high_experiment == "Greedy":
            best_f=train_Y_HF.max().item() if maximize else train_Y_HF.min().item()
            EI_HF = ExpectedImprovement(model=hf_model, best_f=best_f, maximize=maximize)
            ei_values = torch.tensor([EI_HF(c.unsqueeze(0)).item() for c in candidates])
            max_idx = torch.argmax(ei_values).item()
        elif high_experiment == "Uncertainty":
            hf_model.eval()
            with torch.no_grad():
                posterior = hf_model.posterior(candidates)
                variance = posterior.variance.squeeze(-1)
            max_idx = torch.argmax(variance).item()
        else:
            raise ValueError(f"Unknown high_experiment type: {high_experiment}")

        for i, candidate in enumerate(candidates):
            candidate = candidate.unsqueeze(0)
            if i == max_idx:
                new_y = high_fidelity_function(candidate).view(-1, 1)
                train_X_HF = torch.cat([train_X_HF, candidate], dim=0)
                train_Y_HF = torch.cat([train_Y_HF, new_y], dim=0)

                train_X = torch.cat([train_X, candidate], dim=0)
                train_Y = torch.cat([train_Y, new_y], dim=0)

                yl = low_fidelity_function(candidate).item()
                high_low.add_point(candidate, yl, new_y.item())
            else:
                new_y = low_fidelity_function(candidate).view(-1, 1)
                # train_X_LF = torch.cat([train_X_LF, candidate.unsqueeze(0)], dim=0)
                # train_Y_LF = torch.cat([train_Y_LF, new_y], dim=0)

                yl = new_y.item()
                yh_est = high_low.estimate_hf(candidate, yl)
                yh_est = torch.tensor([[yh_est]], dtype=train_Y.dtype, device=train_Y.device)

                train_X = torch.cat([train_X, candidate], dim=0)
                train_Y = torch.cat([train_Y, yh_est], dim=0)

    return pd.DataFrame(rows)