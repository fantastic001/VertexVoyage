
import os 
import yaml
import argparse
import sys
import scipy as sc
from scipy import stats as S 
import numpy as np
import pandas as pd
import re 

parser = argparse.ArgumentParser(description="Analyze outputs from different iterations")
parser.add_argument("spec_file", help="Specification file for the analysis")
parser.add_argument("input_dir", help="Directory containing the output files to analyze")
args = parser.parse_args()

spec = yaml.safe_load(open(args.spec_file))

groups = spec.get("groups", [])
patterns = spec.get("patterns", [])

inputs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, f))]

def get_iterations(group_input: str):
    return [int(f.split(".")[0]) for f in os.listdir(group_input) if f.endswith(".out")]

def analyze_iteration(group_input: str, iteration: int, patterns: list):
    f = os.path.join(group_input, f"{iteration}.out")
    results = {}
    for pattern in patterns:
        pattern["regex"] = pattern["regex"].replace("\\f", "\\d+(?:\.\\d+)?") # Replace \f with a regex for floating point numbers
        with open(f, "r") as file:
            content = file.read()
            for line in content.splitlines():
                matches = re.findall(pattern["regex"], line)
                if matches:
                    for name, value in zip(pattern["names"], list(matches[0])):
                        results[name] = value
    return results

results = []
for i in inputs:
    input_group_name_mapping = {}
    for g, value in zip(groups, os.path.basename(i).split("-")):
        input_group_name_mapping[g] = value
    iterations = get_iterations(i)
    for iteration in iterations:
        iteration_results = analyze_iteration(i, iteration, patterns)
        results.append({
            "group": input_group_name_mapping,
            "iteration": iteration,
            "results": iteration_results
        })
        print(f"Group: {input_group_name_mapping}, Iteration: {iteration}, Results: {iteration_results}")

# Now, calculate summary statistics for each group and pattern
summary = {}
for entry in results:
    group = tuple(entry["group"].items())
    if group not in summary:
        summary[group] = {}
    for name, value in entry["results"].items():
        if name not in summary[group]:
            summary[group][name] = []
        summary[group][name].append(float(value))

# Convert lists to summary statistics (mean, std)
for group, patterns in summary.items():
    for name, values in patterns.items():
        summary[group][name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }

if len(groups) == 0:
    print("No results to summarize.")
elif len(groups) == 1:
    for group, patterns in summary.items():
        data = {}
        data[groups[0]] = group
        for name, stats in patterns.items():
            data[name] = "%f +- %f" % (stats["mean"], stats["std"])
        df = pd.DataFrame([data])
        print(df.to_markdown())
else:
    print("Summary for all groups:")
    data_list = []
    for group, patterns in summary.items():
        data = {}
        for g, value in group:
            data[g] = value
        for name, stats in patterns.items():
            data[name] = "%f +- %f" % (stats["mean"], stats["std"])
        data_list.append(data)
    df = pd.DataFrame(data_list)
    print(df.to_markdown())

for pattern_name in list(next(iter(summary.values())).keys()):
    print(f"\n\n# Comparing groups for pattern: {pattern_name}\n\n")
    data = []
    for group1, patterns1 in summary.items():
        for group2, patterns2 in summary.items():
            group1_name = ", ".join("%s = %s" % (k, v) for k, v in group1)
            group2_name = ", ".join("%s = %s" % (k, v) for k, v in group2)
            entry_name = "(%s) vs (%s)" % (group1_name, group2_name)
            if group1 >= group2:
                continue
            if pattern_name in patterns1 and pattern_name in patterns2:
                values1 = patterns1[pattern_name]["values"]
                values2 = patterns2[pattern_name]["values"]
                if len(values1) > 1 and len(values2) > 1:
                    stat, p = S.mannwhitneyu(values1, values2)
                    cohen_d = (np.mean(values1) - np.mean(values2)) / np.sqrt((np.std(values1) ** 2 + np.std(values2) ** 2) / 2)
                    
                    data.append({
                        "groups": entry_name,
                        "mannwhitneyu_stat": stat,
                        "mannwhitneyu_p": p,
                        "cohen_d": cohen_d
                    })
                else:
                    data.append(
                        {
                            "groups": entry_name,
                            "mannwhitneyu_stat": None,
                            "mannwhitneyu_p": None,
                            "cohen_d": None
                        }
                    )
    df = pd.DataFrame(data)
    print(df.to_markdown())