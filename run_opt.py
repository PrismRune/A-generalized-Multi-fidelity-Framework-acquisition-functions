from functions import *
from optimization import *

import pandas as pd
from filelock import FileLock
import sys

# Import your optimization functions and Battery class
# from your_module import Battery, run_optimization_high_fidelity, run_optimization_multi_fidelity

if __name__ == "__main__":
    SEED = int(sys.argv[1])
    HIGH_BUDGET = 40
    INITIAL_SAMPLES = 10
    LOW_PER_HIGH = 5
    output_file = "branin.csv"
    lock = FileLock(f"{output_file}.lock")
    acquisition_functions = ['EI', 'PI', 'UCB', 'KG']

    func = Branin()

    for acquisition_function in acquisition_functions:
        print(f"Running acquisition function: {acquisition_function} with seed {SEED}")

        high_ = run_optimization_high_fidelity(
            seed=SEED,
            name=func.name,
            high_budget=HIGH_BUDGET,
            bounds=func.normalized_bounds,
            high_fidelity_function=func.high_fidelity,
            initial_samples=INITIAL_SAMPLES,
            dimensions=func.dimension,
            acquisition_function=acquisition_function,
            maximize=True
        )

        multi_ = run_optimization_multi_fidelity(
            seed=SEED,
            name=func.name,
            high_budget=HIGH_BUDGET,
            bounds=func.normalized_bounds,
            high_fidelity_function=func.high_fidelity,
            low_fidelity_function=func.low_fidelity,
            initial_samples=INITIAL_SAMPLES,
            dimensions=func.dimension,
            low_per_high=LOW_PER_HIGH,
            acquisition_function=acquisition_function,
            high_experiment="Greedy",
            maximize=True
        )

        multi2_ = run_optimization_multi_fidelity(
            seed=SEED,
            name=func.name,
            high_budget=HIGH_BUDGET,
            bounds=func.normalized_bounds,
            high_fidelity_function=func.high_fidelity,
            low_fidelity_function=func.low_fidelity,
            initial_samples=INITIAL_SAMPLES,
            dimensions=func.dimension,
            low_per_high=LOW_PER_HIGH,
            acquisition_function=acquisition_function,
            high_experiment="Uncertainty",
            maximize=True
        )

        # Safe append to CSV
        with lock:
            high_.to_csv(output_file, mode="a", index=False, header=False)
            multi_.to_csv(output_file, mode="a", index=False, header=False)
            multi2_.to_csv(output_file, mode="a", index=False, header=False)

