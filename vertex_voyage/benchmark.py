import sys
from types import SimpleNamespace
import pandas as pd
from typing import Callable, List, Dict

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

def data_profile(solvers: dict[str, callable], focus_function: List[str], problems: List[callable]):
    """
    This function performs Wild and More data profile. 
    
    :param solvers: Dict of solvers to profile. Solver is a callable with one parameter - problem.
    :param focus_function: Name of the function to focus on. This function will be patched.
    :param problems: List of problems to solve. Problem is a callable with no parameters.
    
    :return: panda.DataFrame with the following columns:
        - problem: The problem that was solved.
        - solver: The solver that was used.
        - call_count: The number of calls to the function.
    """
    data = []
    for problem in problems:
        for solver_name, solver in solvers.items():
            mock = CallCountingMock(focus_function)
            with SimplePatch(focus_function, mock):
                solver(problem())
            data.append({
                'problem': problem.__name__,
                'solver': solver_name,
                'call_count': mock.call_count
            })
    return pd.DataFrame(data)
   


if __name__ == "__main__":
    import sys
    import math

    # count number of calls to math.pow
    mock = CallCountingMock("math.pow")

    my_solver = lambda x: math.pow(x, 2)
    problems = [lambda: 2, lambda: 3]
    print(data_profile({"my_solver": my_solver}, "math.pow", problems))