class Benchmark:
    def run(self, results_path):
        raise NotImplementedError("Subclasses should implement this method.")
    def display(self, results_path):
        raise NotImplementedError("Subclasses should implement this method.")
