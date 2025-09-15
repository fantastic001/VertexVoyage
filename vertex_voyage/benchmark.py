import sys
from types import SimpleNamespace
import pandas as pd
from typing import Callable, List, Dict
from vertex_voyage.config import get_classes_inheriting, get_config_str
import os.path
from vertex_voyage.benchmark_base import Benchmark
import matplotlib.pyplot as plt
from hashlib import sha256
import inspect 
from vertex_voyage.stats import run_with_monitoring

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

def data_profile(solvers: dict[str, callable], focus_function, problems: List[callable], benchmark: Benchmark = None):
    """
    This function performs Wild and More data profile. 
    
    :param solvers: Dict of solvers to profile. Solver is a callable with one parameter - problem.
    :param focus_function: Name of the function to focus on. This function will be patched. Otherwise, tuple of class and attr name can be passed.
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
            if isinstance(focus_function, str):
                mock = CallCountingMock(focus_function)
                with SimplePatch(focus_function, mock):
                    solver(problem())
                data.append({
                    'problem': problem.__name__,
                    'solver': solver_name,
                    'call_count': mock.call_count
                })
            else:
                old_func = getattr(focus_function[0], focus_function[1])
                mock = CallCountingMock(old_func)
                setattr(focus_function[0], focus_function[1], mock)
                solver(problem())
                setattr(focus_function[0], focus_function[1], old_func)
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

def get_benchmark_results_folder(name: str):
    """
    Get the results folder for a benchmark class.
    
    :param name: The name of the benchmark class.
    :return: The results folder for the benchmark class.
    """
    results_folder = get_config_str("results_folder", "results/", "Path to the results folder.")
    return os.path.join(results_folder, name)

def benchmark_changed(benchmark_class):
    """
    Checks if benchmark run method has been changed by checking hash file in results folder.
    
    :param benchmark_class: The benchmark class.
    :return: True if the benchmark run method has been changed, False otherwise.
    """
    name = get_benchmark_name(benchmark_class)
    results_folder = get_benchmark_results_folder(name)
    hash_file = os.path.join(results_folder, "hash.txt")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        save_benchmark_hash(benchmark_class)
        return True
    if os.path.exists(os.path.join(results_folder, "error.txt")):
        print(f"Results folder {results_folder} contains error.txt file, benchmark run method has been changed.")
        os.remove(os.path.join(results_folder, "error.txt"))
        return True
    if not os.path.exists(hash_file):
        save_benchmark_hash(benchmark_class)
        return False 
    with open(hash_file, "r") as f:
        hash_value = f.read()
    new_hash_value = save_benchmark_hash(benchmark_class)
    if hash_value != new_hash_value:
        return True
    return False

def save_benchmark_hash(benchmark_class):
    """
    Save the hash of the benchmark class to a file.
    
    :param benchmark_class: The benchmark class.
    """
    name = get_benchmark_name(benchmark_class)
    results_folder = get_benchmark_results_folder(name)
    hash_file = os.path.join(results_folder, "hash.txt")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # if source code exists, use it to create hash
    source = inspect.getsourcefile(benchmark_class.run)
    if source:
        with open(source, "r") as f:
            code = f.read()
        # get the line number of the run method
        start_line = benchmark_class.run.__code__.co_firstlineno
        end_line = start_line
        lines = code.splitlines()
        count_tabs = lambda line: len(line) - len(line.lstrip("\t").lstrip(" "))
        indentation = count_tabs(lines[start_line])
        for line in lines[start_line:]:
            if count_tabs(line) < indentation:
                break
            end_line += 1
        code = lines[start_line:end_line]
        code = "\n".join(code)
        code = code.encode('utf-8')
    else:
        code = benchmark_class.run.__code__.co_code
        vars = benchmark_class.run.__code__.co_varnames
        vats = list(sorted(vars))
        vars = " ".join(vats)    
        code = str(code) + str(vars)
        code = code.encode('utf-8')
    # we also add VV environment to the hash 
    for k in sorted(os.environ.keys()):
        if k.startswith("VERTEX_VOYAGE_"):
            code += f"{k}={os.environ[k]}".encode('utf-8')
    hash_value = sha256(code).hexdigest()
    with open(hash_file, "w") as f:
        f.write(hash_value)
    return hash_value

def run_benchmark(name: str, display: bool = True, check_changed: bool = True):
    """
    Run a benchmark by name.
    
    :param name: The name of the benchmark class.
    :param display: Whether to display the benchmark results.
    :param check_changed: Whether to check if the benchmark has changed.
    :return: The result of the benchmark.
    """
    cls = get_benchmark_class(name)
    benchmark = cls()
    results_folder = get_config_str("results_folder", "results/", "Path to the results folder.")
    results_folder = get_benchmark_results_folder(name)
    if check_changed:
        if benchmark_changed(cls):
            benchmark.run(results_folder)
        if display:
            benchmark.display(results_folder)
    else:
        if display:
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
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactively select benchmark")
    parser.add_argument("--no-display", action="store_true", help="Do not display the benchmark results.")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks.")
    parser.add_argument("--no-check-changed", action="store_true", help="Do not check if the benchmark has changed.")
    args = parser.parse_args()
    if args.list:
        print("Available benchmarks:")
        for name in get_benchmark_names():
            print(f" - {name}")
        sys.exit(0)
    if args.interactive:
        print("Available benchmarks:")
        benchmarks = get_benchmark_classes()
        for i, b in enumerate(benchmarks):
            print(f"{i+1}: {get_benchmark_name(b)}")
        i = int(input("> "))
        run_with_monitoring(run_benchmark,get_benchmark_name(benchmarks[i-1]), not args.no_display)
        sys.exit(0)
    if args.all:
        print("Running all benchmarks...")
        all_names = get_benchmark_names()
        all_names = sorted(all_names, key=lambda x: x.lower())
        if not all_names:
            print("No benchmarks found.")
            sys.exit(0)
        print(f"Found {len(all_names)} benchmarks.")
        for i, name in enumerate(all_names):
            print(f"Running benchmark {i+1}/{len(get_benchmark_names())}: {name}...")
            try:
                run_with_monitoring(
                    run_benchmark,
                    name, 
                    not args.no_display, 
                    not args.no_check_changed
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error running benchmark {name}: {e}")
                print("Exception traceback printed above.")
                if os.path.exists(get_benchmark_results_folder(name)):
                    print(f"Results folder {get_benchmark_results_folder(name)} exists, but benchmark failed. Please check the results folder for more information.")
                    error_file = os.path.join(get_benchmark_results_folder(name), "error.txt")
                    with open(error_file, "w") as f:
                        f.write(str(e))
                else:
                    print(f"Results folder {get_benchmark_results_folder(name)} does not exist, benchmark failed. Please check the reexception traceback above for more information.")
        sys.exit(0)
    if args.benchmark:
        print(f"Running benchmark {args.benchmark}...")
        result = run_with_monitoring(
            run_benchmark, 
            args.benchmark, 
            not args.no_display
        )
        print(result)
    else:
        print("Running all benchmarks...")
        for name in get_benchmark_names():
            print(f"Running benchmark {name}...")
            result = run_with_monitoring(
                run_benchmark, 
                name, 
                not args.no_display, 
                not args.no_check_changed
            )
            print(result)
    

