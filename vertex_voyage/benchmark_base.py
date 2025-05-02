class Benchmark:
    def run(self, results_path):
        raise NotImplementedError("Subclasses should implement this method.")
    def display(self, results_path):
        raise NotImplementedError("Subclasses should implement this method.")

    def report_progress(self, current, total):
        """
        Report the progress of the benchmark.
        
        :param current: The current progress.
        :param total: The total progress.
        """
        if total == 0:
            return 
        bar = "#" * int((current / total) * 20)
        bar += "-" * (20 - len(bar))
        bar = f" [{bar}] "
        print(f"Progress: ({(current/total)*100:.2f}%)" + bar, end="\r")