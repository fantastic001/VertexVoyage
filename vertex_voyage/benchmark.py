import sys
from types import SimpleNamespace
import pandas as pd
from typing import Callable, List, Dict
from vertex_voyage.config import get_classes_inheriting, get_config_str
import os.path
from vertex_voyage.benchmark_base import Benchmark
import matplotlib.pyplot as plt
class SimplePatch:
    def __init__(self, target_path, new_value=None):
        self.target_path = target_path
        self.new_value = new_value
        self.original_value = None

    def _resolve(self):
        # Split module and attribute
        module_path, attr_name = self.target_path.rsplit('.', 1)
        module = sys.modules[module_path]
        return module, attr_name

    def __enter__(self):
        module, attr_name = self._resolve()
        self.original_value = getattr(module, attr_name)
        setattr(module, attr_name, self.new_value)
        return self.new_value

    def __exit__(self, exc_type, exc_val, exc_tb):
        module, attr_name = self._resolve()
        setattr(module, attr_name, self.original_value)

# Usage:
# Suppose module_a has a function foo we want to patch
# import module_a
# with SimplePatch('module_a.foo', lambda: 42):
#     assert module_a.foo() == 42

def simple_patch_decorator(target_path, new_value=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with SimplePatch(target_path, new_value):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class CallCountingMock:
    def __init__(self, original_func):
        if isinstance(original_func, str):
            original_func = sys.modules[".".join(original_func.split('.')[:-1])].__dict__[original_func.split('.')[-1]]
        self.original_func = original_func
        self.call_count = 0
        self.call_args = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args.append((args, kwargs))
        return self.original_func(*args, **kwargs)

def data_profile(solvers: dict[str, callable], focus_function: List[str], problems: List[callable], benchmark: Benchmark = None):
    """
    This function performs Wild and More data profile. 
    
    :param solvers: Dict of solvers to profile. Solver is a callable with one parameter - problem.
    :param focus_function: Name of the function to focus on. This function will be patched.
    :param problems: List of problems to solve. Problem is a callable with no parameters.
    :param benchmark: The benchmark object. This is used to report the progress.
    
    :return: panda.DataFrame with the following columns:
        - problem: The problem that was solved.
        - solver: The solver that was used.
        - call_count: The number of calls to the function.
    """
    data = []
    total = len(problems) * len(solvers)
    current = 0
    for problem in problems:
        for solver_name, solver in solvers.items():
            current += 1
            mock = CallCountingMock(focus_function)
            with SimplePatch(focus_function, mock):
                solver(problem())
            data.append({
                'problem': problem.__name__,
                'solver': solver_name,
                'call_count': mock.call_count
            })
            if benchmark:
                benchmark.report_progress(current, total)
    return pd.DataFrame(data)

def display_data_profile(df: pd.DataFrame):
    """
    Display the data profile results.
    
    :param df: The data profile DataFrame.
    """
    solvers = df['solver'].unique()
    alpha_values = range(0, df['call_count'].max() + 1)
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

def get_benchmark_name(benchmark_class):
    """
    Get the name of the benchmark class.
    
    :param benchmark_class: The benchmark class.
    :return: The name of the benchmark class.
    """
    return benchmark_class.NAME if hasattr(benchmark_class, 'NAME') else benchmark_class.__name__
def get_benchmark_classes():
    """
    Get all benchmark classes in the currently loaded plugins.
    :return: A list of benchmark classes.
    """
    return get_classes_inheriting(Benchmark)
def get_benchmark_class(name: str):
    """
    Get a benchmark class by name.
    return: The benchmark class.
    """
    classes = get_benchmark_classes()
    for cls in classes:
        if get_benchmark_name(cls) == name:
            return cls
    raise ValueError(f"Benchmark class {name} not found.")



def run_benchmark(name: str):
    """
    Run a benchmark by name.
    
    :param name: The name of the benchmark class.
    :return: The result of the benchmark.
    """
    cls = get_benchmark_class(name)
    benchmark = cls()
    results_folder = get_config_str("results_folder", "results/", "Path to the results folder.")
    results_folder = os.path.join(results_folder, name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        benchmark.run(results_folder)
    benchmark.display(results_folder)

def get_benchmark_names():
    """
    Get all benchmark names.
    
    :return: A list of benchmark names.
    """
    classes = get_benchmark_classes()
    return [get_benchmark_name(cls) for cls in classes]



if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run a benchmarks")
    parser.add_argument("benchmark", type=str, help="The name of the benchmark to run.", default=None, nargs="?")
    parser.add_argument("--list", action="store_true", help="List all available benchmarks.")
    
    args = parser.parse_args()
    if args.list:
        print("Available benchmarks:")
        for name in get_benchmark_names():
            print(f" - {name}")
        sys.exit(0)
    if args.benchmark:
        print(f"Running benchmark {args.benchmark}...")
        result = run_benchmark(args.benchmark)
        print(result)
    else:
        print("Running all benchmarks...")
        for name in get_benchmark_names():
            print(f"Running benchmark {name}...")
            result = run_benchmark(name)
            print(result)
    

