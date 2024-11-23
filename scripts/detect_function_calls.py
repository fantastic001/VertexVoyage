"""
This script read python file and writes to the standard output all parameters for every
function call of a specified function name patterns.
"""

import ast
import sys
import argparse
from typing import List, Dict, Any, Tuple
import re 
import pandas as pd 
import os 

def print_code(node):
    """
    Print code of a node
    """
    if isinstance(node, ast.Constant):
        return node.value
    return ast.unparse(node)

def detect_calls(filepath):
    """
    Detects function calls in a line of code and their parameters
    """
    content = open(filepath).read()
    calls = []
    try:
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        exit(1)
    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    call = {
                        "name": node.func.attr,
                        "args": []
                    }
                elif isinstance(node.func, ast.Name):
                    call = {
                        "name": node.func.id,
                        "args": []
                    }
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        call["args"].append(arg.value)
                    elif isinstance(arg, ast.Name):
                        call["args"].append(arg.id)
                    elif isinstance(arg, ast.List):
                        call["args"].append([print_code(elt) for elt in arg.elts])
                    elif isinstance(arg, ast.Dict):
                        call["args"].append({str(k): str(v) for k, v in zip(arg.keys, arg.values)})
                    else:
                        call["args"].append(None)
                calls.append(call)
        except Exception as e:
            lineno = node.lineno if hasattr(node, "lineno") else -1
            print(f"Error parsing {filepath} at line {lineno}: {e}")
            exit(1)
    return calls

def get_function_calls(file_path: str, function_name_patterns: List[str]) -> Dict[str, List[Tuple[str, List[Any]]]]:
    """
    Get all function calls with their parameters for a given list of function names
    """
    function_calls = {}
    calls = detect_calls(file_path)
    for call in calls:
        for pattern in function_name_patterns:
            if re.match(pattern, call["name"]):
                if call["name"] not in function_calls:
                    function_calls[call["name"]] = []
                function_calls[call["name"]].append((file_path, call["args"]))
    return function_calls

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Get function calls with their parameters")
    parser.add_argument("file", help="Python file to analyze")
    parser.add_argument("function_names", nargs="+", help="Function names to search for")
    parser.add_argument("--exclude", nargs="+", help="Function names to exclude", default=[])
    args = parser.parse_args()
    if os.path.isfile(args.file):
        function_calls = get_function_calls(args.file, args.function_names)
    elif os.path.isdir(args.file):
        function_calls = {}
        for root, _, files in os.walk(args.file):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    fc = get_function_calls(file_path, args.function_names)
                    for name, calls in fc.items():
                        if args.exclude:
                            if name not in args.exclude:
                                if name not in function_calls:
                                    function_calls[name] = []
                                function_calls[name].extend(calls)
    else:
        print(f"Invalid file or directory: {args.file}")
    parameters = []
    for name, calls in function_calls.items():
        param_type = name.replace("get_config_", "")
        for call in calls:
            try:
                file, arguments = call
                param = arguments[0]
                default_value = arguments[1]
                doc = arguments[2]
                if param in [p["param"] for p in parameters]:
                    print(f"Parameter {param} defined multiple times.")
                    exit(1)
                parameters.append({
                    "param": param,
                    "type": param_type,
                    "default": default_value,
                    "doc": doc
                })
            except Exception as e:
                print(f"Error parsing {file} and function {name}: {e}")
                exit(1)
    df = pd.DataFrame(parameters)
    print(df.to_markdown(index=False))



if __name__ == "__main__":
    main()

    