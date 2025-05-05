
from vertex_voyage.temporal import to_nx_graph, ForestFireEventSequence, FirstN, animate_graph
from vertex_voyage.benchmark_base import Benchmark
import matplotlib.pyplot as plt
import pandas as pd 
import os 

class ForestFireBenchmark(Benchmark):
    """
    Forest Fire Benchmark for evaluating temporal graph algorithms.
    """
    NAME = "Forest Fire Degree Distribution"

    def run(self, results_path):
        """
        Run the Forest Fire benchmark and save the results.
        """
        # Generate a forest fire event sequence
        event_sequence = FirstN(ForestFireEventSequence(.1), 1000)
        events = list(event_sequence)
        g = to_nx_graph(events)
        # Compute the degree distribution
        degree_sequence = [d for n, d in g.degree()]
        data = [] 
        for d in degree_sequence:
            data.append((d, 1))
        df = pd.DataFrame(data, columns=["degree", "count"])
        df = df.groupby("degree").sum().reset_index()
        df["degree"] = df["degree"].astype(int)
        df.to_csv(os.path.join(results_path, "forest_fire_degree_distribution.csv"), index=False)
    
    def display(self, results_path):
        """
        Display the results of the Forest Fire benchmark.
        """
        try:
            # Load the degree distribution data
            df = pd.read_csv(os.path.join(results_path, "forest_fire_degree_distribution.csv"))
            # Plot the degree distribution
            plt.figure(figsize=(10, 6))
            plt.bar(df["degree"], df["count"], width=0.8, color='blue')
            plt.xlabel("Degree")
            plt.ylabel("Count")
            plt.title("Forest Fire Degree Distribution")
            plt.grid(True)
            plt.show()

            
        except FileNotFoundError:
            self.run(results_path)
            self.display(results_path)
