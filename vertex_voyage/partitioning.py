

import networkx as nx 
from cdlib.algorithms import lfm
from cdlib.algorithms.internal.lfm import Community
from binpacking import to_constant_bin_number
import random 
import vertex_voyage.config as cfg 

def modified__lfm(G: nx.Graph, partition_count, alpha: float = 1,threshold: float = 0.5) -> list:
    communities = []
    node_not_include = list(G.nodes.keys())[:]
    node_num = len(node_not_include)
    while len(node_not_include) > node_num * threshold:
        c = Community(G, alpha)
        # randomly select a seed node
        seed = random.choice(node_not_include)
        c.add_node(seed)

        to_be_examined = c.get_neighbors()
        while to_be_examined:
            # largest fitness to be added
            m = {}
            for node in to_be_examined:
                fitness = c.cal_add_fitness(node)
                m[node] = fitness
            to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

            # stop condition
            if to_be_add[1] < 0.0:
                break
            c.add_node(to_be_add[0])

            to_be_remove = c.recalculate()
            while to_be_remove is not None:
                c.remove_vertex(to_be_remove)
                to_be_remove = c.recalculate()

            to_be_examined = c.get_neighbors()

        for node in c.nodes:
            if node in node_not_include:
                node_not_include.remove(node)
        communities.append(list(c.nodes))
    if len(communities) < partition_count:
        for _ in range(partition_count - len(communities)):
            communities.append(list())
    for node in node_not_include:
        random_comm = random.choice(communities)
        random_comm.append(node)
    return list(communities)

@cfg.pluggable
def partition_graph(G: nx.Graph, partition_num: int, use_modified_lfm: bool = False, threshold: float = 0.5, alpha: float = 1) -> list:
    """
    Partition the graph into a given number of partitions using LFM algorithm.
    """
    # create a LFM object
    communities = None 
    if use_modified_lfm:
        communities = modified__lfm(G, partition_num, alpha=alpha, threshold=threshold)
    else:
        communities = lfm(G, alpha=alpha).communities
    # partition the graph into a given number of partitions
    partitions = to_constant_bin_number(communities, partition_num, key=len)
    partitions = [list(sum(part, [])) for part in partitions]
    return partitions

@cfg.pluggable
def calculate_partitioning_corruption(G: nx.Graph, partitions: list):
    """
    Partitioning corruption is 1 - ratio of size of edges of union of subgraphs of G induced by partitions and number of edges in original graph G  
    """
    # calculate the number of edges in original graph G
    original_edges = G.number_of_edges()
    # calculate the number of edges in union of subgraphs of G induced by partitions
    partitions_edges = set()
    for partition in partitions:
        subgraph = G.subgraph(partition)
        partitions_edges ^= set(subgraph.edges)
    # calculate the partitioning corruption
    partitioning_corruption = 1-len(partitions_edges) / original_edges
    return partitioning_corruption

@cfg.pluggable
def calculate_corruptability(G: nx.Graph, partition_num: int, use_modified_lfm = False, threshold = 0.5, alpha: float = 1, partitions = None):
    """
    Calculate the corruptability of the graph.
    """
    # partition the graph into a given number of partitions
    if partitions is None:
        partitions = partition_graph(G, partition_num, use_modified_lfm=use_modified_lfm, threshold=threshold, alpha=alpha)
    # calculate the partitioning corruption
    corruption = calculate_partitioning_corruption(G, partitions)
    return corruption


def evaluate_partition_sbm(G, partitions):
    """
    Evaluate how a given partition compares to a ground-truth partition for an SBM graph,
    returning *ratios* instead of raw counts.

    Parameters
    ----------
    G : networkx.Graph
        The generated (SBM) graph.
    ground_truth : dict
        Dictionary {node: community_label}.
    partitions : list of lists
        Each element is a list of nodes in a particular partition.

    Returns
    -------
    ratio_intra_lost : float
        Ratio of intra-community edges that were 'lost' (i.e., placed in different partitions)
        compared to the total number of intra-community edges.
    ratio_inter_separated : float
        Ratio of inter-community edges that are separated (i.e., endpoints in different partitions)
        compared to the total number of inter-community edges.
    """
    ground_truth = {}
    for block_label, nodes in enumerate(G.graph["partition"]):
        for node in nodes:
            ground_truth[node] = block_label
    # 1) Create a map from node -> index of the partition
    node_to_partition = {}
    for partition_index, part_nodes in enumerate(partitions):
        for n in part_nodes:
            node_to_partition[n] = partition_index
    # 2) Counters for edges
    same_comm_diff_part = 0  # # of intra-community edges in different partitions
    diff_comm_diff_part = 0  # # of inter-community edges in different partitions
    total_intra_edges = 0    # total # of intra-community edges
    total_inter_edges = 0    # total # of inter-community edges

    # 3) Go through each edge in the graph
    for u, v in G.edges():
        # Are u and v in the same ground-truth community?
        same_community = (ground_truth[u] == ground_truth[v])

        # Are u and v in the same or different partitions?
        same_partition = (node_to_partition[u] == node_to_partition[v])

        if same_community:
            # This is an intra-community edge
            total_intra_edges += 1
            if not same_partition:
                same_comm_diff_part += 1  # Lost inside different partitions
        else:
            # This is an inter-community edge
            total_inter_edges += 1
            if not same_partition:
                diff_comm_diff_part += 1  # Correctly separated (different partitions)
    print("Intra-community edges: ", total_intra_edges)
    print("Same community different partition: ", same_comm_diff_part)
    print("Inter-community edges: ", total_inter_edges)
    print("Different community different partition: ", diff_comm_diff_part)
    # 4) Compute ratios (handling possible division by zero)
    ratio_intra_lost = (
        same_comm_diff_part / total_intra_edges if total_intra_edges > 0 else 0
    )
    ratio_inter_separated = (
        diff_comm_diff_part / total_inter_edges if total_inter_edges > 0 else 0
    )
    return ratio_intra_lost, ratio_inter_separated

def get_performance_profile(solvers, problems):
    datapoints = [] 
    total = len(problems) * len(solvers)
    progress = 0
    for pi, p in enumerate(problems):
        for si, s in enumerate(solvers):
            progress += 1
            print(f"Progress: {progress}/{total}", end="\r")
            datapoints.append(({
                "problem": pi,
                "solver": si, 
                "metric": s(p())
            }))
    print()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df = pd.DataFrame(datapoints)
    best = df.groupby("problem")["metric"].transform("min") + 1e-9
    df["ratio"] = df["metric"] / best
    solvers = list(range(len(solvers)))
    n_problems = len(problems)
    tau_values = np.logspace(0, 3, 100)
    plt.figure()
    for i, solver in enumerate(solvers):
        tau = []
        for t in tau_values:
            tau.append((df[df["solver"] == solver]["ratio"] <= t).sum() / n_problems)
        plt.plot(tau_values, tau, label=f"S{i}")
    plt.xscale("log")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Fraction of problems solved")
    plt.legend()
    plt.grid()
    plt.show()
    return df

@cfg.pluggable
def calculate_graph_corruptability(G: nx.Graph, max_partition_num: int, use_modified_lfm = False, threshold=0.5, alpha: float = 1):
    """
    Calculate the corruptability of the graph for all partition numbers from 1 to max_partition_num and returns linear coefficient of function corruptability(pnum) = k * (pnum-1).
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    # calculate the corruptability of the graph for all partition numbers from 1 to max_partition_num
    x = np.array(range(1, max_partition_num+1)).reshape(-1, 1)-1
    y = np.array([calculate_corruptability(G, pnum, use_modified_lfm=use_modified_lfm, threshold=threshold, alpha=alpha) for pnum in range(1, max_partition_num+1)]).reshape(-1, 1)
    # fit the linear regression model
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0]

def random_partitioning(G: nx.Graph, partition_num: int):
    """
    Randomly partition the graph into a given number of partitions.
    """
    nodes = list(G.nodes)
    random.shuffle(nodes)
    partitions = to_constant_bin_number(nodes, partition_num, key= lambda x: 1)
    return partitions

if __name__ == "__main__":
    import vertex_voyage.config as cfg
    import numpy as np
    import networkx as nx
    stage = cfg.get_config_int("pm_stage", 0, "Stage in main partitioning measurement script")
    be_fast = cfg.get_config_bool("pm_be_fast", False, "Whether to be fast in main partitioning measurement script")
    fast_factor = cfg.get_config_int("pm_fast_factor", 100, "Fast factor in main partitioning measurement script")
    G = nx.karate_club_graph()
    problems = []
    def solver_corruptability(alpha, threshold):
        return lambda G: np.mean([calculate_graph_corruptability(G, 10, use_modified_lfm=True, threshold=threshold, alpha=alpha) for k in range(30)])
    def solver_inter_loss(alpha, threshold, partitioner = None):
        def f(G):
            if partitioner is None:
                partitions = partition_graph(G, 10, use_modified_lfm=True, threshold=threshold, alpha=alpha)
            else:
                partitions = partitioner(G, 10)
            return evaluate_partition_sbm(G, partitions)[1]
        return f
    def solver_intra_loss(alpha, threshold, partitioner = None):
        def f(G):
            if partitioner is None:
                partitions = partition_graph(G, 10, use_modified_lfm=True, threshold=threshold, alpha=alpha)
            else:
                partitions = partitioner(G, 10)
            return evaluate_partition_sbm(G, partitions)[0]
        return f
    solvers = [
        solver_corruptability(1, 0),
        solver_corruptability(1, 0.5),
        solver_corruptability(1, 1),
    ]
    # add several common graphs into problems
    problems += [
        lambda: nx.karate_club_graph(),
        lambda: nx.florentine_families_graph(),
        lambda: nx.les_miserables_graph()
    ]
    # # add several BA graphs
    # for n in range(10, 1000, 100):
    #     for m in range(1, 10):
    #         problems.append(lambda: nx.barabasi_albert_graph(n, int(0.1*m*n)))
    # # add several ER graphs
    # for n in range(10, 1000, 100):
    #     for p in range(1, 10):
    #         problems.append(lambda: nx.erdos_renyi_graph(n, 0.1*p))
    # # add several WS graphs
    # for n in range(10, 1000, 100):
    #     for k in range(1, 10):
    #         for p in range(1, 10):
    #             problems.append(lambda: nx.watts_strogatz_graph(n, k, 0.1*p))
    # # Add several SBM graphs
    # for n in range(10, 1000, 100):
    #     for k in range(1, 10):
    #         for p in range(1, 10):
    #             problems.append(lambda: nx.planted_partition_graph(n, k, 0.1*p, 0.1*p))
    if stage == 0 or stage == 1:
        df = get_performance_profile(solvers, random.sample(problems, fast_factor) if be_fast else problems)
        print(df)
    problems = [] 
    solvers = [
        solver_inter_loss(1, 0),
        solver_inter_loss(1, 0.5),
        solver_inter_loss(1, 1),
    ]

    for n in range(2, 11):
        for k in range(1,10):
            for p in range(5, 10):
                for q in range(1, 5):
                    problems.append(lambda: nx.planted_partition_graph(n, k, 0.1*p, 0.01*q))
    if stage == 0 or stage == 2:
        df = get_performance_profile(solvers, random.sample(problems, fast_factor) if be_fast else problems)
        print(df)
    solvers = [
        solver_inter_loss(1, 0, random_partitioning),
        solver_inter_loss(1, 0.5)
    ]
    if stage == 0 or stage == 3:
        df = get_performance_profile(solvers, random.sample(problems, fast_factor) if be_fast else problems)
    if stage == 0 or stage == 4:
        threshold_values = [] 
        inter_community_loss = []
        intra_community_loss = []
        data = []
        for threshold in range(0, 11):
            threshold_values.append(threshold/10)
            inter = [] 
            intra = []
            for problem in (random.sample(problems, fast_factor) if be_fast else problems):
                G = problem()
                partitions = partition_graph(G, 5, use_modified_lfm=True, threshold=threshold/10, alpha=1)
                x,y = evaluate_partition_sbm(G, partitions)
                intra.append(x)
                inter.append(y)
            inter_community_loss.append(np.mean(inter))
            intra_community_loss.append(np.mean(intra))
            data.append({
                "threshold": threshold/10,
                "inter_community_loss": np.mean(inter),
                "intra_community_loss": np.mean(intra)
            })
        import matplotlib.pyplot as plt
        plt.plot(threshold_values, inter_community_loss, label="Inter-community loss")
        plt.plot(threshold_values, intra_community_loss, label="Intra-community loss")
        plt.xlabel("Threshold")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        import pandas as pd
        print(pd.DataFrame(data))