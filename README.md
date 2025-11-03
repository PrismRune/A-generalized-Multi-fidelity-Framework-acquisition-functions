# A Generalized Multi-Fidelity Framework

A concise framework for **multi-fidelity Bayesian optimization** on synthetic benchmarks: **Forrester, Branin, Borehole, Hartmann**.

## Features

* High- and multi-fidelity optimization.
* Supports acquisition functions: `EI`, `PI`, `UCB`, `KG`.
* High-fidelity selection strategies: `Greedy` or `Uncertainty`.
* Safe concurrent CSV writing.
* Modular for adding custom functions or acquisition strategies.

## Installation

```bash
git clone https://github.com/PrismRune/A-generalized-Multi-fidelity-Framework-acquisition-functions
cd A-generalized-Multi-fidelity-Framework-acquisition-functions
pip install -r requirements.txt
```

## Usage

Run from command line:

```bash
python run_opt.py <SEED>
```

You can also see example usage and how to run the optimization functions in `run_opt.ipynb`.

Parameters:

* `SEED`: random seed
* `HIGH_BUDGET`, `INITIAL_SAMPLES`, `LOW_PER_HIGH`: optimization budgets
* `acquisition_functions`: list of acquisition functions
* `high_experiment`: `Greedy` or `Uncertainty`
* `output_file`: CSV for results

Example usage in code:

```python
from functions import *
from optimization import *

func = Branin()
results = run_optimization_multi_fidelity(
    seed=42,
    name=func.name,
    high_budget=40,
    bounds=func.normalized_bounds,
    high_fidelity_function=func.high_fidelity,
    low_fidelity_function=func.low_fidelity,
    initial_samples=10,
    dimensions=func.dimension,
    low_per_high=5,
    acquisition_function='KG',
    high_experiment='Greedy',
    maximize=True
)
```

## Extending

* Add new acquisition functions or synthetic benchmark functions.
* Adjust budgets or high/low fidelity ratios.
* Set `high_experiment` to control high-fidelity selection strategy.
