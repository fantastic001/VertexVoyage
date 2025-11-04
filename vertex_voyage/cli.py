
import sys
import numpy as np
from vertex_voyage.grid_search import (
    GridSearchPersistence, 
    grid_search, 
    last, 
    identity
)
from vertex_voyage.command_executor import (
    command_executor_main, 
    command_executor_rpc
)
from vertex_voyage.command_executor import get_classes
from vertex_voyage.partitioning import (
    calculate_partitioning_corruption,
    get_partition_average_balance,
)
from experiments.datasets import datasets
from vertex_voyage.temporal import to_nx_graph, to_vv_graph, Transform, Event
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.distger import DistGER
from vertex_voyage.reconstruction import get_f1_score, reconstruct
from vertex_voyage.partitioning import (
    partition_graph,
    label_propagation_partitioner
)
from datetime import datetime
from vertex_voyage.config import get_config_str


GS_LOCATION = get_config_str("gs_cache_location", "gs_cache", "Location to store grid search results")

def perform_embedding(cls, alg: str, p: float, q: float, dim: int, dataset_name: str, part_num: int):
    gsp = GridSearchPersistence(GS_LOCATION)
    print("Processing dataset ", dataset_name)
    t = VertexEnumerator()
    dataset = Transform(datasets[dataset_name](), lambda x: Event(
        src=t(int(x.src)),
        dest=t(int(x.dest)),
        timestamp=int(x.timestamp),
        type=x.type,
        attrs=x.attrs,
    ))
    dataset = to_vv_graph(dataset)
    if part_num == 1:
        gsp["dataset"] = dataset_name
        gsp["num_parts"] = 1
        gsp["alpha"] = 0
        gsp["threshold"] = 0
        gsp["p"] = p
        gsp["q"] = q
        gsp["dim"] = dim
        log("Embedding full graph...")
        model = cls()
        model.fit(dataset)
        embedding = model.embed_nodes(dataset.nodes)
        gsp.save([embedding], algorithm=alg)
    else:
        for params, partitions in gsp.load(
            dataset=dataset_name, 
            num=part_num
        ):
            result = [] 
            model = cls(
                p=p,
                q=q,
                dim=dim
            )
            gsp["dataset"] = dataset_name
            gsp["num_parts"] = params["num"]
            gsp["partitioning_algorithm"] = params.get("algorithm", "unknown")
            gsp["alpha"] = params["alpha"]
            gsp["threshold"] = params["threshold"]
            gsp["p"] = p
            gsp["q"] = q
            gsp["dim"] = dim
            log("Embedding partitions...")
            log(f"   Partitions: {len(partitions)}")
            log(f"  Dataset: {dataset_name}")
            log(f"  Alpha: {params['alpha']}")
            log(f"  Num parts: {params['num']}")
            log(f"  Threshold: {params['threshold']}")
            for part in partitions:
                part = [t(x) for x in part]
                model.fit(dataset.subgraph(part), dataset.nodes)
                embedding = model.embed_nodes(part)
                result.append(embedding)
            gsp.save(result, algorithm=alg)
class VertexEnumerator:
    def __init__(self):
        self.visited = set()
        self.index = {}
    
    def __call__(self, node):
        if node not in self.visited:
            self.visited.add(node)
            self.index[node] = len(self.visited) - 1
        return self.index[node]


def log(*messages):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(map(str, messages))}")
    sys.stdout.flush()


class Commands:
    def serve(self):
        from vertex_voyage.cluster import register_node, get_binding_port
        from vertex_voyage.config import notify_plugins, get_config_int
        from vertex_voyage import ControlInterface
        notify_plugins("node_starting")
        register_node()
        notify_plugins("node_started")
        notify_plugins("register_commands", ControlInterface())
        command_executor_rpc(get_classes("vertex_voyage") + ControlInterface.additional_classes, get_binding_port())
    def list(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        results = []
        for h, p in gsp.list():
            ps = ', '.join(f"{k}={v}" for k, v in p.items())
            results.append(f"{h}: {ps}")
        return results

    def list_datasets(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        datasets = set()
        for params, result in gsp.load(threshold=0):
            datasets.add(params['dataset'])
        return list(datasets)
    
    def restore(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        gsp.restore_backups()

    def graph_corruptability(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        result = [] 
        for dataset_name in self.list_datasets():
            print("Processing dataset ", dataset_name)
            dataset = to_nx_graph(datasets[dataset_name]())
            for params, res in gsp.load(threshold=0, dataset=dataset_name):
                part_num = params["num"]

                c = calculate_partitioning_corruption(dataset, res)
                result.append({
                    'dataset': dataset_name,
                    'num_parts': part_num,
                    "alpha": params["alpha"],
                    'corruptability': c
                })
        return result
    
    def node2vec(self, dataset_name: str, part_num: int, p: float, q: float, dim: int):
        return perform_embedding(Node2Vec, 'node2vec', p, q, dim, dataset_name, part_num)
    def distger(self, dataset_name: str, part_num: int, p: float, q: float, dim: int):
        return perform_embedding(DistGER, 'distger', p, q, dim, dataset_name, part_num)

    def score(self, algorithm: str, dataset_name: str, num_parts: int, alpha: float, threshold: float):
        gsp = GridSearchPersistence(GS_LOCATION)
        scores = []
        for params, embeddings in gsp.load(
            algorithm=algorithm,
            dataset=dataset_name,
            num_parts=num_parts,
            alpha=alpha,
            threshold=threshold
        ):
            log(f"Scoring embeddings for dataset {dataset_name} with params: {params}")
            t = VertexEnumerator()
            dataset = Transform(datasets[dataset_name](), lambda x: Event(
                src=t(int(x.src)),
                dest=t(int(x.dest)),
                timestamp=int(x.timestamp),
                type=x.type,
                attrs=x.attrs,
            ))
            dataset = to_nx_graph(dataset)
            if num_parts == 1:
                parts = [list(dataset.nodes)]
            else:
                parts = gsp.load(
                    dataset=dataset_name,
                    num=num_parts,
                    alpha=alpha,
                    threshold=threshold
                )
                assert len(parts) == 1
                parts = parts[0][1]
                parts = [[t(x) for x in part] for part in parts]
            mapping = {} 
            assert len(parts) == len(embeddings)
            assert all(len(part) == len(embedding) for part, embedding in zip(parts, embeddings))
            for part, embedding in zip(parts, embeddings):
                for node, vec in zip(part, embedding):
                    mapping.setdefault(node, []).append(vec)
            for node in mapping:
                mapping[node] = np.mean(mapping[node], axis=0)
            embeddings = [mapping[node] for node in dataset.nodes]
            assert len(embeddings) == dataset.number_of_nodes()
            assert all(emb is not None for emb in embeddings)
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert all(isinstance(x, int) for x in dataset.nodes)
            reconstructed = reconstruct(dataset.number_of_edges(), embeddings)
            f1_score = get_f1_score(dataset, reconstructed)
            # calculate ARI 
            from sklearn.cluster import KMeans
            single_embeddings = gsp.load(
                algorithm=algorithm,
                dataset=dataset_name,
                num_parts=1,
                p=params['p'],
                q=params['q'],
                dim=params['dim']
            )
            assert len(single_embeddings) == 1
            single_embeddings = single_embeddings[0][1][0]
            ari_data = {}
            for clusters in [1,2,3]:
                single_k_means = KMeans(n_clusters=clusters).fit_predict(single_embeddings)
                k_means = KMeans(n_clusters=clusters).fit_predict(embeddings)
                from sklearn.metrics import adjusted_rand_score
                similarity = adjusted_rand_score(single_k_means, k_means)
                ari_data[f'ari_{clusters}'] = similarity
            scores.append({
                'f1_score': f1_score,
                'dataset': dataset_name,
                'num_parts': num_parts,
                'alpha': alpha,
                'threshold': threshold,
                **ari_data
            })
        return scores
    
    def score_partitioning(self, dataset_name: str, num_parts: int, algorithm: str):
        gsp = GridSearchPersistence(GS_LOCATION)
        scores = []
        for params, partitions in gsp.load(
            dataset=dataset_name,
            num=num_parts,
            algorithm=algorithm
        ):
            log(f"Scoring partitioning for dataset {dataset_name} with params: {params}")
            dataset = to_nx_graph(datasets[dataset_name]())
            c = calculate_partitioning_corruption(dataset, partitions)
            balance = get_partition_average_balance({i: len(p) for i, p in enumerate(partitions)}, len(partitions))
            scores.append({
                'corruptibility': c,
                'dataset': dataset_name,
                'balance': balance,
                'num_parts': num_parts,
                'alpha': params['alpha'],
                'threshold': params['threshold']
            })
        return scores

    def lfm(self, dataset_name: str):
        gs_persist = GridSearchPersistence(GS_LOCATION)
        whitelist = [
            dataset_name
        ] 
        datasets_ = {k: v for k, v in datasets.items() if k in whitelist}
        for dataset_name, dataset in datasets_.items():
            gs_persist['dataset'] = dataset_name
            gs_persist['algorithm'] = 'lfm'
            g = dataset()
            g = to_vv_graph(g)
            log(f"Dataset: {dataset_name}")
            log(f"   Number of nodes: {g.number_of_nodes()}")
            mp = grid_search(
                f = lambda threshold, alpha, num: partition_graph(
                    alpha=alpha,
                    threshold=threshold,
                    G=g,
                    partition_num=num,
                    use_modified_lfm=True
                ),
                apply=identity,
                acc=last,
                param_ranges={
                    'threshold': [0, .5, 1],
                    'alpha': [1,2, 3],
                    'num': [2,4,8,16]
                }, 
                intermediate_callback=gs_persist,
                report_progress=True
            )
            log(f"\nMinimum corruptability: {mp}")

    def lpa(self, dataset_name: str):
        gsp = GridSearchPersistence(GS_LOCATION)
        gsp["algorithm"] = "lpa"
        dataset = to_nx_graph(datasets[dataset_name]())
        grid_search(
            f = lambda num: label_propagation_partitioner(
                G=dataset,
                partition_num=num
            ),
            apply=identity,
            acc=last,
            param_ranges={
                'num': [2,4,8,16]
            },
            intermediate_callback=gsp,
            report_progress=True
        )
    
    def fix(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        gsp.restore()
    

    def remove_dataset(self, dataset_name: str):
        gsp = GridSearchPersistence(GS_LOCATION)
        gsp.delete(dataset=dataset_name)

    def test(self, *, 
             name: str = "CITESEER", 
             partitions: int = 2, 
             alpha: float = 1.0, 
             threshold: float = 0.0
    ):

        import networkx as nx
        gsp = GridSearchPersistence(GS_LOCATION)
        log("Processing dataset ")
        t = VertexEnumerator()
        dataset = Transform(datasets[name](), lambda x: Event(
            src=t(int(x.src)),
            dest=t(int(x.dest)),
            timestamp=int(x.timestamp),
            type=x.type,
            attrs=x.attrs,
        ))
        dataset = to_nx_graph(dataset)
        parts = partition_graph(dataset, partitions, alpha=alpha, threshold=threshold, use_modified_lfm=True)
        log("Total number of nodes: ", dataset.number_of_nodes())
        log("Graph partitioned")
        embs = {}
        for part in parts:
            log("Partition size: %d" % len(part))
            best = None
            best_f1 = -1
            pg = dataset.subgraph(part)
            gg = nx.Graph()
            gg.add_edges_from(pg.edges)
            cs = nx.connected_components(gg)
            cs = list(reversed(sorted(cs, key=len)))
            log("Biggest components: ", [len(x) for x in cs[:3]])
            log("Isolated nodes: ", len(list(nx.isolates(gg))))
            log("Number of connected components: ", nx.number_connected_components(gg))
            log("Degree distribution: ", nx.degree_histogram(gg)[:5])
            log("Average clustering: ", nx.average_clustering(gg))
            log("Partition number of edges: ", pg.number_of_edges())
            all_nodes = list(dataset.nodes)
            for p in [0.25, 0.5, 1, 2, 4]:
                for q in [0.25, 0.5, 1, 2, 4]:
                    model = Node2Vec(dim=100, p=p, q=q, n_walks=1, walk_size=10, window_size=3)
                    model.fit(pg, dataset.nodes)
                    emb = model.embed_nodes(part)
                    g = reconstruct(pg.number_of_edges(), emb, part)
                    PG = nx.Graph()
                    PG.add_edges_from(pg.edges)
                    f1 = get_f1_score(PG, g)
                    if f1 > best_f1:
                        best_f1 = f1
                        best = emb
                        log("New best: ", p, q, best_f1)
            log("Best achieved F1 score: ", best_f1)
            for node, e in zip(part, best):
                if node not in embs:
                    embs[node] = []
                embs[node].append(e)
        for n in dataset.nodes:
            embs[n] = np.mean(embs[n], axis=0)
        embs = [embs[n] for n in dataset.nodes]
        g = reconstruct(dataset.number_of_edges(), embs, list(dataset.nodes))
        G = nx.Graph()
        G.add_edges_from(dataset.edges)
        log("Global F1 score: ", get_f1_score(G, g))



if __name__ == "__main__":
    command_executor_main(Commands)
