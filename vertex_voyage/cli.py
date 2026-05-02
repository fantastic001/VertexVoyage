
import random
import sys
import numpy as np
from vertex_voyage import data, dynnode2vec
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
    calculate_corruptability,
    calculate_partitioning_corruption,
    get_partition_average_balance,
)
from experiments.datasets import datasets, dataset_params
from vertex_voyage.temporal import EventStream, batched, to_nx_graph, to_vv_graph, Transform, Event
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.distger import DistGER
from vertex_voyage.reconstruction import get_f1_score, reconstruct
from vertex_voyage.partitioning import (
    partition_graph,
    label_propagation_partitioner
)
from datetime import datetime
from vertex_voyage.config import get_config_str

from vertex_voyage.dynnode2vec import DynNode2Vec

from vertex_voyage.temporal_partitioning import WindowedLabelPropagationTemporalGraphPartitioner

import logging 

logger = logging.getLogger("CLI")

def setup_logging():
    logging.basicConfig(
        level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }[get_config_str(
            "log_level", "INFO", 
            "Log level for the VV (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
        ).upper()],
        format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename=get_config_str(
            "log_file", "vv.log", "Log file for the VV"
        )
    )

GS_LOCATION = get_config_str("gs_cache_location", "gs_cache", "Location to store grid search results")

ALGS = {
    "node2vec": Node2Vec,
    "distger": DistGER,
}

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
    logger.info(' '.join(map(str, messages)))


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
             threshold: float = 0.0,
             break_early: bool = False,
             skip_global: bool = False,
             dim: int = 100,
             default_p: float = 0,
             default_q: float = 0,
             epochs: int = 10,
             long_run : bool = False,
             use_dataset_params: bool = False,
             use_lpa: bool = False,
             algorithm: str = "node2vec"
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
        if not use_lpa:
            parts = partition_graph(dataset, partitions, alpha=alpha, threshold=threshold, use_modified_lfm=True)
        else:
            parts = label_propagation_partitioner(dataset, partitions)
        log("Total number of nodes: ", dataset.number_of_nodes())
        log("Graph partitioned")
        embs = {}
        for part in parts:
            log("Partition size: %d" % len(part))
            if len(part) == 0:
                print("Skipping empty partition")
                continue
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
            alg = ALGS[algorithm]
            for p in [0.25, 0.5, 1, 2, 4]:
                for q in [0.25, 0.5, 1, 2, 4]:
                    if ((default_p > 0 and default_q > 0) and 
                        not (p == default_p and q == default_q)):
                        continue
                    if long_run:
                        n_walks = 10
                        walk_size = 80
                        window_size = 10
                    else:
                        n_walks = 1
                        walk_size = 10
                        window_size = 3
                    if use_dataset_params:
                        params: dict = dataset_params.get(name, {})
                        n_walks = params.get('n_walks', n_walks)
                        walk_size = params.get('walk_size', walk_size)
                        window_size = params.get('window_size', window_size)
                        epochs = params.get('epochs', epochs)
                        dim = params.get('dim', dim)
                        p = params.get('p', p)
                        q = params.get('q', q)
                    if alg == DistGER:
                        P = {
                            "min_walk_size": walk_size // 2,
                            "max_walk_size": walk_size * 2,
                        }
                    else:
                        P = {
                            "walk_size": walk_size,
                        }
                    model = alg(
                        p=p,
                        q=q,
                        dim=dim,
                        n_walks=n_walks,
                        window_size=window_size,
                        epochs=epochs,
                        **P
                    )
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
                    if break_early:
                        break
                if break_early:
                    break
            log("Best achieved F1 score: ", best_f1)
            for node, e in zip(part, best):
                if node not in embs:
                    embs[node] = []
                embs[node].append(e)
        if skip_global:
            log("Skipping global F1 computation")
            return
        for n in dataset.nodes:
            embs[n] = np.mean(embs[n], axis=0)
        embs = [embs[n] for n in dataset.nodes]
        g = reconstruct(dataset.number_of_edges(), embs, list(dataset.nodes))
        G = nx.Graph()
        G.add_edges_from(dataset.edges)
        log("Global F1 score: ", get_f1_score(G, g))
    
    def evaluate_partitioning(
            self, 
            name: str, 
            *,
            threshold: float = 0.0,
            alpha: float = 1.0,
            use_lpa: bool = False
        ):
        dataset = to_nx_graph(datasets[name]())
        log("Processing dataset ", name)
        if use_lpa:
            parts = label_propagation_partitioner(dataset, partition_num=4)
        else:
            parts = partition_graph(dataset, partition_num=4, alpha=alpha, threshold=threshold, use_modified_lfm=True)
        c = calculate_partitioning_corruption(dataset, parts)
        balance = get_partition_average_balance({i: len(p) for i, p in enumerate(parts)}, len(parts))
        log(f"Dataset: {name}")
        log(f"   Number of nodes: {dataset.number_of_nodes()}")
        log(f"   Corruptibility: {c}")
        log(f"   Balance: {balance}")
    
    def temporal_test(self, *, 
             name: str = "CITESEER", 
             partitions: int = 1, 
             alpha: float = 1.0, 
             threshold: float = 0.0,
             break_early: bool = False,
             skip_global: bool = False,
             dim: int = 100,
             default_p: float = 0,
             default_q: float = 0,
             epochs: int = 1,
             long_run : bool = False,
             use_dataset_params: bool = False,
             use_lpa: bool = False,
             algorithm: str = "node2vec",
             track_seen: bool = False,
             iterations: int = 1,
             limit: int = -1,
             batch_size: int = 100
    ):

        import networkx as nx
        scores = []
        t = VertexEnumerator()
        dataset = Transform(datasets[name](), lambda x: Event(
            src=t(int(x.src)),
            dest=t(int(x.dest)),
            timestamp=int(x.timestamp),
            type=x.type,
            attrs=x.attrs,
        ))
        if limit > 0:
            og_events = list(dataset)[:limit]
        else:
            og_events = list(dataset)
        original_graph = to_nx_graph(og_events)
        logger.debug(f"Original graph has {original_graph.number_of_nodes()} nodes and {original_graph.number_of_edges()} edges")
        if use_dataset_params:
            params: dict = dataset_params.get(name, {})
            alpha = params.get('alpha', alpha)
            threshold = params.get('threshold', threshold)
            dim = params.get('dim', dim)
            default_p = params.get('p', default_p)
            default_q = params.get('q', default_q)
        models = [DynNode2Vec(
            dim=dim,
            epochs=epochs,
            p= default_p if default_p > 0 else 0.5,
            q= default_q if default_q > 0 else 0.5,
            n_walks=10 if long_run else 1,
            walk_size=80 if long_run else 10,
            window_size=10 if long_run else 3,
            retrain_threshold=int(0.1 * original_graph.number_of_nodes())
        ) for _ in range(partitions)]
        for it in range(iterations):
            log(f"Iteration {it+1} / {iterations}: Processing dataset {name}")
            events = og_events.copy()
            if track_seen:
                random.shuffle(events)
            nodes = set()
            seen = set()
            i = 0
            old_f1_score = 0
            sorted_events = []
            while(len(events) > 0):
                seen_status = "maybe seen"
                if track_seen:
                    # find event with smallest timestamp among events that are connected to seen nodes
                    event = None
                    for e in events:
                        if e.src in seen or e.dest in seen:
                            event = e
                            seen_status = "seen"
                            break
                    if event is None:
                        event = events[0]
                        seen_status = "not seen"
                    events.remove(event)
                    seen.add(event.src)
                    seen.add(event.dest)
                else:
                    event = events.pop(0)
                log(f"Processing {seen_status} event {i+1} / timestamp {event.timestamp}")
                nodes.add(event.src)
                nodes.add(event.dest)
                sorted_events.append(event)
            total_edges = 0
            for bi, batch in enumerate(batched(sorted_events, batch_size)):
                total_edges += len(batch)
                models_to_update = set([0])
                for model_id in models_to_update:
                    models[model_id].update(batch)
                    embeddings = models[0].embed_nodes(nodes)
                # reconstruct graph and compute F1 score
                g = reconstruct(total_edges, embeddings, list(nodes))
                G = nx.Graph()
                for u,v in original_graph.edges:
                    if u in nodes and v in nodes:
                        G.add_edge(u, v)
                try:
                    f1_score = get_f1_score(G, g)
                except ZeroDivisionError:
                    f1_score = 0.0
                log(f"Batch: {bi+1}, F1 score: {f1_score}")
                if old_f1_score > 0 and f1_score < old_f1_score * 0.5:
                    logger.warn(f"F1 score dropped significantly from {old_f1_score} to {f1_score} at batch {bi+1}")
                old_f1_score = f1_score
            log("Event stream processing completed")
            scores.append(old_f1_score)
        log("Average F1 score: ", np.mean(scores))
        log("Standard deviation of F1 score: ", np.std(scores))
        node2vec = Node2Vec(
            dim=dim, 
            p=default_p if default_p > 0 else 0.5, 
            q=default_q if default_q > 0 else 0.5,
            n_walks=10 if long_run else 1,
            walk_size=80 if long_run else 10,
            window_size=10 if long_run else 3,
            epochs=epochs, 
        )
        # Fit on the full graph for an upper bound on performance
        node2vec.fit(original_graph, original_graph.nodes)
        # Compute F1 score for the full graph        full_emb =
        full_emb = node2vec.embed_nodes(original_graph.nodes)
        full_g = reconstruct(original_graph.number_of_edges(), full_emb, list(original_graph.nodes))
        full_f1_score = get_f1_score(original_graph, full_g)
        log("F1 score for full graph using Node2Vec: ", full_f1_score)
        return 
        dataset = to_nx_graph(dataset)
        if not use_lpa:
            parts = partition_graph(dataset, partitions, alpha=alpha, threshold=threshold, use_modified_lfm=True)
        else:
            parts = label_propagation_partitioner(dataset, partitions)
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
            alg = ALGS[algorithm]
            for p in [0.25, 0.5, 1, 2, 4]:
                for q in [0.25, 0.5, 1, 2, 4]:
                    if ((default_p > 0 and default_q > 0) and 
                        not (p == default_p and q == default_q)):
                        continue
                    if long_run:
                        n_walks = 10
                        walk_size = 80
                        window_size = 10
                    else:
                        n_walks = 1
                        walk_size = 10
                        window_size = 3
                    if use_dataset_params:
                        params: dict = dataset_params.get(name, {})
                        n_walks = params.get('n_walks', n_walks)
                        walk_size = params.get('walk_size', walk_size)
                        window_size = params.get('window_size', window_size)
                        epochs = params.get('epochs', epochs)
                        dim = params.get('dim', dim)
                        p = params.get('p', p)
                        q = params.get('q', q)
                    if alg == DistGER:
                        P = {
                            "min_walk_size": walk_size // 2,
                            "max_walk_size": walk_size * 2,
                        }
                    else:
                        P = {
                            "walk_size": walk_size,
                        }
                    model = alg(
                        p=p,
                        q=q,
                        dim=dim,
                        n_walks=n_walks,
                        window_size=window_size,
                        epochs=epochs,
                        **P
                    )
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
                    if break_early:
                        break
                if break_early:
                    break
            log("Best achieved F1 score: ", best_f1)
            for node, e in zip(part, best):
                if node not in embs:
                    embs[node] = []
                embs[node].append(e)
        if skip_global:
            log("Skipping global F1 computation")
            return
        for n in dataset.nodes:
            embs[n] = np.mean(embs[n], axis=0)
        embs = [embs[n] for n in dataset.nodes]
        g = reconstruct(dataset.number_of_edges(), embs, list(dataset.nodes))
        G = nx.Graph()
        G.add_edges_from(dataset.edges)
        log("Global F1 score: ", get_f1_score(G, g))



if __name__ == "__main__":
    command_executor_main(Commands)
