
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
