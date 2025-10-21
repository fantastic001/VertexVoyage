
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
                model.fit(dataset.subgraph(part), dataset.nodes)
                embedding = model.embed_nodes([t(x) for x in part])
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


def log(message: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
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
        return list(k for k, _ in gsp.load())
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
            scores.append({
                'f1_score': f1_score,
                'dataset': dataset_name,
                'num_parts': num_parts,
                'alpha': alpha,
                'threshold': threshold
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
        

if __name__ == "__main__":
    command_executor_main(Commands)