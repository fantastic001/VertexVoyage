
from vertex_voyage import node2vec
from vertex_voyage.grid_search import GridSearchPersistence
from vertex_voyage.command_executor import command_executor_main
from vertex_voyage.partitioning import calculate_partitioning_corruption
from experiments.datasets import datasets
from vertex_voyage.temporal import to_nx_graph, to_vv_graph, Transform, Event
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.reconstruction import get_f1_score, reconstruct

GS_LOCATION = "gs_cache"

class VertexEnumerator:
    def __init__(self):
        self.visited = set()
        self.index = {}
    
    def __call__(self, node):
        if node not in self.visited:
            self.visited.add(node)
            self.index[node] = len(self.visited) - 1
        return self.index[node]


class Commands:
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
    
    def embedding(self, dataset_name: str, part_num: int):
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
        for params, partitions in gsp.load(dataset=dataset_name, num=part_num):
            result = [] 
            model = Node2Vec()
            gsp["dataset"] = dataset_name
            gsp["num_parts"] = params["num"]
            gsp["alpha"] = params["alpha"]
            gsp["threshold"] = params["threshold"]
            if params["num"] <= 4:
                continue
            print("Embedding partitions...")
            print("   Partitions: ", len(partitions))
            print("  Dataset: ", dataset_name)
            print("  Alpha: ", params["alpha"])
            print("  Num parts: ", params["num"])
            print("  Threshold: ", params["threshold"])
            for part in partitions:
                model.fit(dataset.subgraph(part), dataset.nodes)
                embedding = model.embed_nodes([t(x) for x in part])
                result.append(embedding)
            gsp.save(result, algorithm="node2vec")

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
            reconstructed = reconstruct(len(dataset.nodes), embeddings, dataset.nodes)
            f1_score = get_f1_score(dataset, embeddings)
            scores.append({
                'f1_score': f1_score,
                'dataset': dataset_name,
                'num_parts': num_parts,
                'alpha': alpha,
                'threshold': threshold
            })
        return scores


if __name__ == "__main__":
    command_executor_main(Commands)