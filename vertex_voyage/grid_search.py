
import json 
import pickle
import os 
from datetime import datetime
import hashlib

from vertex_voyage.temporal import FirstN

def grid_search(
        f, 
        apply, 
        acc, 
        param_ranges, 
        fixed_params=None, 
        intermediate_callback=None,
        report_progress=False
    ):
    """
    Perform a grid search over specified parameter ranges and calculate accumulated results.
    
    Formally, result is as follows:

    $$ \text{result} = \bigoplus_{p_1 \in P_1} \bigoplus_{p_2 \in P_2} ... \bigoplus_{p_n \in P_n} \alpha(f(p_1, p_2, ..., p_n, \text{fixed\_params})) $$

    where \( P_i \) is the set of values for parameter \( p_i \), $\alpha$ is the apply function, and \( \bigoplus \) is the accumulation operation.

    Parameters:
    f : function
        The function to evaluate, which takes parameters and returns a result.
    apply : function
        A function that processes the output of `f`.
    acc : function
        A binary function that accumulates results.
    param_ranges : dict
        A dictionary where keys are parameter names and values are lists of parameter values.
    fixed_params : dict, optional
        A dictionary of parameters that remain constant during the search.
    intermediate_callback : function, optional
        A function that is called with the current result after each evaluation of `f` where current parameters are passed to callback as kwargs.
    report_progress : bool, optional
        If True, progress will be reported.

    Returns:
    The accumulated result after applying `f` over all combinations of parameters in `param_ranges`.
    """
    from itertools import product

    if fixed_params is None:
        fixed_params = {}

    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    result = None
    P = product(*param_values)
    total = 1
    i = 0
    if report_progress:
        P = list(P)
        total = len(P)
        i = 0
    for values in P:
        params = dict(zip(param_names, values))
        params.update(fixed_params)
        output = f(**params)
        if intermediate_callback is not None:
            intermediate_callback(output, **params)
        applied_output = apply(output)
        
        if result is None:
            result = applied_output
        else:
            result = acc(result, applied_output)
        if report_progress:
            i += 1
            print(f"Progress: " + "=" * (i * 20 // total) + " " * (20 - (i * 20 // total)) + f" {i}/{total}", end='\r')

    return result


class GridSearchPersistence:
    """
    A class to handle saving and loading the state of a grid search.
    """

    def __init__(self, location):
        """
        Initialize the persistence handler.

        Results are saved into specified directory. Every result is child directory with its identifier which is hash of the parameters used for that run.

        Each directory contains a file `state.pkl` which contains the pickled state of the grid search at that point.

        Along with state file, a `params.json` file is saved which contains the parameters used for that particular run.

        Parameters:
        location : str
            The directory where intermediate results will be saved.
        """
        self.location = location
        self.fixed_params = {}
    

    def __setitem__(self, key, value):
        self.fixed_params[key] = value
    def __getitem__(self, key):
        return self.fixed_params[key]
    def __delitem__(self, key):
        del self.fixed_params[key]
    def __contains__(self, key):
        return key in self.fixed_params


    def save(self, state, **params):
        """
        Save the current state and parameters to disk.

        If there is old state already saved, it will be backed up by renaming the directory to `<id>_timestamp`. The timestamp is in the format `YYYYMMDD_HHMMSS`.
        """
        if not os.path.exists(self.location):
            os.makedirs(self.location)

        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()

        dirpath = os.path.join(self.location, param_hash)
        if os.path.exists(dirpath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.location, f"{param_hash}_{timestamp}")
            os.rename(dirpath, backup_path)
        
        os.makedirs(dirpath, exist_ok=True)

        state_path = os.path.join(dirpath, 'state.pkl')
        params_path = os.path.join(dirpath, 'params.json')

        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        with open(params_path, 'w') as f:
            p = {}
            p.update(self.fixed_params)
            p.update(params)
            json.dump(p, f, indent=4, default=str)
        
        
    def load(self, **params) -> list:
        """
        Load the state corresponding to the given parameters.

        Parameters:
        params : dict
            The parameters used to identify the saved state.

        Returns:
        The list of tuples (params, result) if found.
        """
        children = os.listdir(self.location)
        results = []
        for child in children:
            child_path = os.path.join(self.location, child)
            if os.path.isdir(child_path):
                # skip backup directories
                if '_' in child:
                    continue
                params_path = os.path.join(child_path, 'params.json')
                state_path = os.path.join(child_path, 'state.pkl')
                if os.path.exists(params_path) and os.path.exists(state_path):
                    with open(params_path, 'r') as f:
                        saved_params = json.load(f)
                    if all(saved_params.get(k) == v for k, v in params.items()):
                        with open(state_path, 'rb') as f:
                            state = pickle.load(f)
                            results.append((saved_params, state))
        return results
    
    def __call__(self, result, **kwargs):
        return self.save(result, **kwargs)

identity = lambda x: x



class no_serialize:
    def __init__(self, obj):
        self.obj = obj
    def __str__(self) -> str:
        return ""
    def __repr__(self) -> str:
        return ""

if __name__ == "__main__":
    from experiments.datasets import datasets
    from vertex_voyage.partitioning import partition_graph, min_corruptability
    from vertex_voyage.vv_graph import VVGraph
    from vertex_voyage.temporal import to_vv_graph
    gs_persist = GridSearchPersistence(location="gs_cache")
    whitelist = [
        "Twitch"
    ] 
    datasets = {k: v for k, v in datasets.items() if k in whitelist}
    for dataset_name, dataset in datasets.items():
        gs_persist['dataset'] = dataset_name
        g = FirstN(dataset(), n=100)
        g = to_vv_graph(g)
        print(f"Dataset: {dataset_name}")
        print("   Number of nodes:", g.number_of_nodes())
        mp = grid_search(
            f = lambda threshold, alpha, num: partition_graph(
                alpha=alpha,
                threshold=threshold,
                G=g,
                partition_num=num,
                use_modified_lfm=True
            ),
            apply=identity,
            acc=lambda p1, p2: min_corruptability(
                g, p1, p2
            ),
            param_ranges={
                'threshold': [0, .5, 1],
                'alpha': [.5, 1,2],
                'num': [2,4,8,16]
            }, 
            intermediate_callback=gs_persist,
            report_progress=True
        )
        print("\nMinimum corruptability:", mp)