
from vertex_voyage.benchmark import data_profile, CallCountingMock, SimplePatch
import numpy as np 
from vertex_voyage.benchmark_base import Benchmark

class SampleBenchmark(Benchmark):
    """
    A sample benchmark class for demonstration purposes.
    """
    NAME = "SampleBenchmark"

    def run(self, results_path):
        import math

        # count number of calls to math.pow
        mock = CallCountingMock("math.pow")

        my_solver = lambda x: math.pow(x, 2)
        problems = [lambda: 2, lambda: 3]
        df = data_profile({"my_solver": my_solver}, "math.pow", problems)
        df.to_csv(f"{results_path}/results.csv", index=False)

    def display(self, results_path):
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(f"{results_path}/results.csv")
        solvers = df['solver'].unique()
        alpha_values = np.arange(0, 201, 10)  # from 0 to 200 in steps of 10
        for solver in solvers:
            subset = df[df['solver'] == solver]
            fractions = [] 
            total_problems = len(subset)
            for alpha in alpha_values:
                solved = subset[subset['call_count'] <= alpha]
                fraction = len(solved) / total_problems
                fractions.append(fraction)
            plt.plot(alpha_values, fractions, label=solver)
        plt.xlabel('Alpha')
        plt.ylabel('Fraction of Problems Solved')
        plt.title('Fraction of Problems Solved vs Alpha')
        plt.legend()
        plt.grid()
        plt.show()