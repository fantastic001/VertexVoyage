
from genericpath import isdir
import pandas as pd 
import os 
import sys 
import matplotlib.pyplot as plt 
import json 

path = sys.argv[1]

images = ["4.3", "4.13", "4.10", "4.4"] 

for file in os.listdir(path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, file))
        basename = ".".join(file.split(".")[:-1])
        if basename not in images:
            print(df.to_latex(index=False, caption=basename, label=f"tab:{basename}"))
        else:
            df = df.set_index(df.columns[0])
            df.plot.bar()
            if os.path.exists(os.path.join(path, f"{basename}.json")):
                with open(os.path.join(path, f"{basename}.json"), 'r') as f:
                    data = json.load(f)
                    for param, value in data.items():
                        getattr(plt, param)(value)
            plt.savefig(os.path.join(path, f"{basename}.png"), bbox_inches='tight')
    if os.path.isdir(os.path.join(path, file)):
        metadata_file = os.path.join(path, file, "metadata.json")
        print("Processing metadata file ", metadata_file)
        metadata = json.load(open(metadata_file, 'r'))
        results_file_name = metadata.get("results_file", "results.csv")
        results_file = os.path.join(path, file, results_file_name) 
        df = pd.read_csv(results_file)
        if "columns" in metadata:
            if isinstance(metadata["columns"], dict):
                df = df[list(metadata["columns"].keys())]
                df = df.rename(columns=metadata["columns"])
            elif isinstance(metadata["columns"], list):
                df = df[metadata["columns"]]
        if "groupby" in metadata:
            df = df.groupby(metadata["groupby"]).mean()
        if "sortby" in metadata:
            df = df.sort_values(by=metadata["sortby"], ascending=True)
        if "plot" in metadata:
            kind = metadata["plot"].get("kind", "bar")
            metadata["plot"].pop("kind", None)
            if kind == "bar":
                df.plot.bar()
                if "plot_params" in metadata:
                    for param, value in metadata["plot_params"].items():
                        getattr(plt, param)(value)
                plt.savefig(os.path.join(path, f"{file}.png"), bbox_inches='tight')
            elif kind == "line":
                df.plot.line()
                if "plot_params" in metadata:
                    for param, value in metadata["plot_params"].items():
                        getattr(plt, param)(value)
                plt.savefig(os.path.join(path, f"{file}.png"), bbox_inches='tight')
            else:
                print(df.to_latex(caption=metadata["plot"]["title"], label=f"tab:{file.replace(' ', '_')}"))