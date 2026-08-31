
from vertex_voyage.cli import (
    ALGS,
    F1_COMPU_THRESHOLD,
    VertexEnumerator,
    hash_set_persistently,
    log,
    _enumerated_event_stream, 
    CustomCLICommandExecutor
)
import vertex_voyage.cli
import random
import logging
import numpy as np

from experiments.datasets import dataset_params
from vertex_voyage.config import notify_plugins
from vertex_voyage.distger import DistGER
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.partitioning import label_propagation_partitioner, partition_graph
from vertex_voyage.persist import PersistedRun
from vertex_voyage.reconstruction import get_f1_score, reconstruct
from vertex_voyage.tasks.link_prediction import (
    evaluate_predictions,
    heart_benchmark,
    train_on_static_graph,
)
from vertex_voyage.temporal import buffered, to_nx_graph
from vertex_voyage.timing import TimeMetric
from vertex_voyage.temporal_partitioning import (
    InMemoryPartition,
    MostCommonNeighborPartitioner,
    Partition,
    PartitionerProfile,
    RandomPartitioner,
)
from vertex_voyage.semantic_temporal_partitioning import (
    SemanticTemporalGraphPartitioner,
)

logger = logging.getLogger("CLI")

class TestCustomCLICommandExecutor(CustomCLICommandExecutor):
    P_Q_GRID = [0.25, 0.5, 1, 2, 4]
    LONG_RUN_WALK_PARAMS = {
        "n_walks": 10,
        "walk_size": 80,
        "window_size": 10,
    }
    SHORT_RUN_WALK_PARAMS = {
        "n_walks": 1,
        "walk_size": 10,
        "window_size": 3,
    }
    LINK_PREDICTION_TEST_FRACTION = 0.1

    def testtest(self):
        return "testtest"

    def _report_timing(self, run):
        """Persist collected TimeMetric aggregates onto the run and log the report."""
        try:
            run["timing"] = TimeMetric.dump()
        except Exception:
            logger.warning("Could not persist timing metrics to run", exc_info=True)
        TimeMetric.print_report()

    def _resolved_default_pq(self, default_p: float, default_q: float):
        return (
            default_p if default_p > 0 else 0.5,
            default_q if default_q > 0 else 0.5,
        )

    def _walk_params(self, long_run: bool):
        if long_run:
            return (
                self.LONG_RUN_WALK_PARAMS["n_walks"],
                self.LONG_RUN_WALK_PARAMS["walk_size"],
                self.LONG_RUN_WALK_PARAMS["window_size"],
            )
        return (
            self.SHORT_RUN_WALK_PARAMS["n_walks"],
            self.SHORT_RUN_WALK_PARAMS["walk_size"],
            self.SHORT_RUN_WALK_PARAMS["window_size"],
        )

    def _apply_dataset_params(self, name: str, n_walks: int, walk_size: int, window_size: int, epochs: int, dim: int, p: float, q: float):
        params: dict = dataset_params.get(name, {})
        return (
            params.get('n_walks', n_walks),
            params.get('walk_size', walk_size),
            params.get('window_size', window_size),
            params.get('epochs', epochs),
            params.get('dim', dim),
            params.get('p', p),
            params.get('q', q),
        )

    def _iter_pq_candidates(self, default_p: float, default_q: float):
        for p in self.P_Q_GRID:
            for q in self.P_Q_GRID:
                if ((default_p > 0 and default_q > 0) and
                    not (p == default_p and q == default_q)):
                    continue
                yield p, q

    def _append_embedding(self, embs: dict, node, embedding):
        if node not in embs:
            embs[node] = []
        embs[node].append(embedding)

    def _initialize_test_dataset(self, run, name: str, link_prediction: bool, t: VertexEnumerator):
        dataset = _enumerated_event_stream(name, t)
        removed_edges = []
        positive_edges = []
        negative_edges = []
        test_edges = []

        if "graph" in run and not link_prediction:
            dataset = run["graph"]
            run["removed_edges"] = []
            run["positive_edges"] = []
            run["negative_edges"] = []
            return dataset, removed_edges, positive_edges, negative_edges, test_edges

        if (
            "graph" in run and
            "removed_edges" in run and
            "positive_edges" in run and
            "negative_edges" in run
        ):
            dataset = run["graph"]
            removed_edges = run["removed_edges"]
            positive_edges = run["positive_edges"]
            negative_edges = run["negative_edges"]
            test_edges = positive_edges + negative_edges
            return dataset, removed_edges, positive_edges, negative_edges, test_edges

        dataset = to_nx_graph(dataset)
        if link_prediction:
            nodes = list(dataset.nodes())
            edges_to_remove = int(self.LINK_PREDICTION_TEST_FRACTION * dataset.number_of_edges())
            removed_edges = list(random.sample(list(dataset.edges()), edges_to_remove))
            positive_edges = removed_edges
            negative_edges = []
            while len(negative_edges) < len(positive_edges):
                u = random.choice(nodes)
                v = random.choice(nodes)
                if not dataset.has_edge(u, v) and u != v:
                    negative_edges.append((u, v))
            test_edges = positive_edges + negative_edges
            dataset.remove_edges_from(removed_edges)
            run["positive_edges"] = positive_edges
            run["negative_edges"] = negative_edges
            run["graph"] = dataset
            run["removed_edges"] = removed_edges
        else:
            run["graph"] = dataset
            run["removed_edges"] = []
            run["positive_edges"] = []
            run["negative_edges"] = []

        return dataset, removed_edges, positive_edges, negative_edges, test_edges

    def _partition_for_test(self, run, dataset, partitions: int, alpha: float, threshold: float, use_lpa: bool):
        if not use_lpa:
            log("Partitioning graph with LFM-based partitioner...")
            return run(
                "partitions",
                partition_graph,
                dataset,
                partitions,
                alpha=alpha,
                threshold=threshold,
                use_modified_lfm=True,
            )

        log("Partitioning graph with label propagation...")
        return run("partitions", label_propagation_partitioner, dataset, partitions)

    def _log_partition_graph_stats(self, nx, pg):
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

    def _load_cached_partition_embedding(self, run, part_name: str, pg, part):
        best = None
        best_f1 = -1
        best_model = None
        if ("model_%s" % part_name) not in run:
            return False, best, best_f1, best_model

        log("Loading model for partition...")
        model = run["model_%s" % part_name]
        log("Model parameters: p=%f, q=%f, dim=%d" % (model.p, model.q, model.dim))
        log("Model loaded, embedding nodes...")
        part = list(part)
        emb = model.embed_nodes(part)
        log("Nodes embedded")
        precision, recall, f1 = get_f1_score(pg, reconstruct(pg.number_of_edges(), emb, part))
        log("Partition scores: Precision: %f, Recall: %f, F1 Score: %f" % (precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            best = emb
            best_model = model
        return True, best, best_f1, best_model

    def _train_best_partition_model(
        self,
        *,
        nx,
        alg,
        name: str,
        pg,
        part,
        dataset,
        default_p: float,
        default_q: float,
        long_run: bool,
        use_dataset_params: bool,
        epochs: int,
        dim: int,
        break_early: bool,
    ):
        best = None
        best_f1 = -1
        best_model = None
        part = list(part)

        for p, q in self._iter_pq_candidates(default_p, default_q):
            n_walks, walk_size, window_size = self._walk_params(long_run)
            local_epochs = epochs
            local_dim = dim
            local_p = p
            local_q = q
            if use_dataset_params:
                (
                    n_walks,
                    walk_size,
                    window_size,
                    local_epochs,
                    local_dim,
                    local_p,
                    local_q,
                ) = self._apply_dataset_params(
                    name,
                    n_walks,
                    walk_size,
                    window_size,
                    local_epochs,
                    local_dim,
                    local_p,
                    local_q,
                )
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
                p=local_p,
                q=local_q,
                dim=local_dim,
                n_walks=n_walks,
                window_size=window_size,
                epochs=local_epochs,
                **P
            )
            model.fit(pg, dataset.nodes)
            emb = model.embed_nodes(part)
            g = reconstruct(pg.number_of_edges(), emb, part)
            PG = nx.Graph()
            PG.add_edges_from(pg.edges)
            precision, recall, f1 = get_f1_score(PG, g)
            if f1 > best_f1:
                best_f1 = f1
                best = emb
                best_model = model
                log("New best: p=%f, q=%f, dim=%d, precision=%f, recall=%f, f1=%f" % (local_p, local_q, local_dim, precision, recall, f1))
            if break_early:
                break

        return best, best_f1, best_model

    @TimeMetric("link_prediction")
    def _run_link_prediction_with_embedding(self, run, dataset, embedding_dict, positive_edges, negative_edges):
        log("Training link prediction model on full graph...")

        class EM:
            def __init__(self, embedding_dict):
                self.embedding_dict = embedding_dict

            def embed_nodes(self, nodes):
                return [self.embed_node(n) for n in nodes]

            def embed_node(self, node):
                return self.embedding_dict[node]

        em = EM(embedding_dict)
        full_model, train_losses, val_losses = run("lp_full", train_on_static_graph, dataset, em, epochs=10)
        log("Full model trained (Train loss: %f, Val loss: %f)" % (train_losses[-1], val_losses[-1]))

        lp_precision, lp_recall, lp_f1, lp_accuracy = run(
            "lp_full_eval",
            evaluate_predictions,
            em,
            full_model,
            positive_edges,
            negative_edges,
        )
        log(f"Full Model - Precision: {lp_precision:.4f}")
        log(f"Full Model - Recall: {lp_recall:.4f}")
        log(f"Full Model - F1 Score: {lp_f1:.4f}")
        log(f"Full Model - Accuracy: {lp_accuracy:.4f}")

        ranks = run("lp_heart_benchmark", heart_benchmark, em, full_model, dataset, positive_edges, ns=500, ps=1000)
        log(f"Full Model - Mean Rank: {ranks.mean_rank():.4f}")
        log(f"Full Model - MRR: {ranks.mrr():.4f}")
        log(f"Full Model - Hits@1: {ranks.hits_at_k(1):.4f}")
        log(f"Full Model - Hits@3: {ranks.hits_at_k(3):.4f}")
        log(f"Full Model - Hits@5: {ranks.hits_at_k(5):.4f}")
        log(f"Full Model - Hits@10: {ranks.hits_at_k(10):.4f}")

    @TimeMetric("create_models")
    def _create_temporal_models(
        self,
        *,
        algorithm: str,
        partitions: int,
        dim: int,
        epochs: int,
        default_p: float,
        default_q: float,
        long_run: bool,
        original_graph,
    ):
        resolved_p, resolved_q = self._resolved_default_pq(default_p, default_q)
        n_walks, walk_size, window_size = self._walk_params(long_run)
        return {
            InMemoryPartition.empty(id=p): ALGS[algorithm](
                dim=dim,
                epochs=epochs,
                p=resolved_p,
                q=resolved_q,
                n_walks=n_walks,
                walk_size=walk_size,
                window_size=window_size,
                retrain_threshold=int(0.1 * original_graph.number_of_nodes())
            ) for p in range(partitions)
        }

    @TimeMetric("create_partitioner")
    def _create_temporal_partitioner(
        self,
        *,
        partitioner_name: str,
        parts,
        replication_factor: int,
        mu: float,
        epsilon: float,
        alpha: float,
        decay: float,
        semantic_metric: str,
        semantic_assignment: str,
        semantic_k: int,
        semantic_eps: float,
        semantic_min_samples: int,
        semantic_reassign_noise: bool,
        semantic_dim: int,
        semantic_epochs: int,
        semantic_p: float,
        semantic_q: float,
        semantic_n_walks: int,
        semantic_walk_size: int,
        semantic_window_size: int,
        semantic_retrain_threshold: int,
    ):
        partitioner = {
            "random": lambda **kw: RandomPartitioner.uniform(parts),
            "random.degree": lambda **kw: RandomPartitioner.degree_based(parts),
            "neighbors.all": lambda **kw: MostCommonNeighborPartitioner.all_neighbors(
                parts,
                replication_factor=kw["replication_factor"],
                mu=kw["mu"],
                epsilon=kw["epsilon"],
                alpha=kw["alpha"],
                decay=kw["decay"]
            ),
            "neighbors.degree": lambda **kw: MostCommonNeighborPartitioner.degree_based(
                parts,
                replication_factor=kw["replication_factor"],
                mu=kw["mu"],
                epsilon=kw["epsilon"],
                alpha=kw["alpha"],
                decay=kw["decay"]
            ),
            "semantic.node2vec": lambda **kw: SemanticTemporalGraphPartitioner.node2vec(
                parts,
                metric=kw["semantic_metric"],
                assignment=kw["semantic_assignment"],
                mu=kw["mu"],
                k=kw["semantic_k"],
                eps=kw["semantic_eps"],
                min_samples=kw["semantic_min_samples"],
                reassign_noise=kw["semantic_reassign_noise"],
                dim=kw["semantic_dim"],
                epochs=kw["semantic_epochs"],
                p=kw["semantic_p"],
                q=kw["semantic_q"],
                n_walks=kw["semantic_n_walks"],
                walk_size=kw["semantic_walk_size"],
                window_size=kw["semantic_window_size"],
            ),
            "semantic.dynnode2vec": lambda **kw: SemanticTemporalGraphPartitioner.dynnode2vec(
                parts,
                metric=kw["semantic_metric"],
                assignment=kw["semantic_assignment"],
                mu=kw["mu"],
                k=kw["semantic_k"],
                eps=kw["semantic_eps"],
                min_samples=kw["semantic_min_samples"],
                reassign_noise=kw["semantic_reassign_noise"],
                dim=kw["semantic_dim"],
                epochs=kw["semantic_epochs"],
                p=kw["semantic_p"],
                q=kw["semantic_q"],
                n_walks=kw["semantic_n_walks"],
                walk_size=kw["semantic_walk_size"],
                window_size=kw["semantic_window_size"],
                retrain_threshold=kw["semantic_retrain_threshold"],
            ),
        }[partitioner_name](
            replication_factor=replication_factor,
            mu=mu,
            epsilon=epsilon,
            alpha=alpha,
            decay=decay if decay > 0 else None,
            semantic_metric=semantic_metric,
            semantic_assignment=semantic_assignment,
            semantic_k=semantic_k,
            semantic_eps=semantic_eps,
            semantic_min_samples=semantic_min_samples,
            semantic_reassign_noise=semantic_reassign_noise,
            semantic_dim=semantic_dim,
            semantic_epochs=semantic_epochs,
            semantic_p=semantic_p,
            semantic_q=semantic_q,
            semantic_n_walks=semantic_n_walks,
            semantic_walk_size=semantic_walk_size,
            semantic_window_size=semantic_window_size,
            semantic_retrain_threshold=semantic_retrain_threshold,
        )
        return PartitionerProfile(partitioner)

    @TimeMetric("sort_events")
    def _sort_temporal_events(self, og_events, track_seen: bool):
        events = og_events.copy()
        if track_seen:
            random.shuffle(events)
        nodes = set()
        seen = set()
        sorted_events = []
        while(len(events) > 0):
            if track_seen:
                event = None
                for e in events:
                    if e.src in seen or e.dest in seen:
                        event = e
                        break
                if event is None:
                    event = events[0]
                events.remove(event)
                seen.add(event.src)
                seen.add(event.dest)
            else:
                event = events.pop(0)
            nodes.add(event.src)
            nodes.add(event.dest)
            sorted_events.append(event)
        return sorted_events

    @TimeMetric("process_buffers")
    def _process_temporal_buffers(
            self, *,
            nx,
            sorted_events, 
            buffer_size: int, 
            partitioner, 
            models, 
            original_graph,
            iteration: int,
            run = None,
        ):
        total_edges = 0
        nodes = set()
        iteration_precisions, iteration_recalls, iteration_f1s = [], [], []
        total_buffers = (len(sorted_events) + buffer_size - 1) // buffer_size
        processed_after_last_f1 = 0
        old_f1_score = 0

        for bi, buffer in enumerate(buffered(sorted_events, buffer_size)):
            processed_after_last_f1 += len(buffer)
            for event in buffer:
                nodes.add(event.src)
                nodes.add(event.dest)
            with TimeMetric("partitioner_push"):
                partitioner.push(buffer)

            total_edges += len(buffer)
            with TimeMetric("model_update"):
                for part, partition_buffer in partitioner.get_partition_buffers(buffer):
                    models[part].update(partition_buffer)

            with TimeMetric("embedding"):
                if run is not None:
                    embeddings = run(f"embedding_{iteration}_buffer_{bi}", partitioner.get_distributed_embedding, models, nodes)
                else:
                    embeddings = partitioner.get_distributed_embedding(models, nodes)

            if processed_after_last_f1 >= F1_COMPU_THRESHOLD or bi == total_buffers - 1 or bi == 0:
                processed_after_last_f1 = 0
                with TimeMetric("reconstruct_and_f1"):
                    g = reconstruct(total_edges, embeddings, list(nodes))
                    G = nx.Graph()
                    for u, v in original_graph.edges:
                        if u in nodes and v in nodes:
                            G.add_edge(u, v)
                    try:
                        precision, recall, f1_score = get_f1_score(G, g)
                    except ZeroDivisionError:
                        precision, recall, f1_score = 0.0, 0.0, 0.0
                log(f"Buffer: {bi+1}/{total_buffers}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}")
                if old_f1_score > 0 and f1_score < old_f1_score * 0.5:
                    logger.warn(f"F1 score dropped significantly from {old_f1_score} to {f1_score} at buffer {bi+1}")
                old_f1_score = f1_score
                iteration_precisions.append(precision)
                iteration_recalls.append(recall)
                iteration_f1s.append(f1_score)
            else:
                log(f"Buffer: {bi+1}/{total_buffers}, F1 score not computed this buffer")

        return old_f1_score, iteration_precisions, iteration_recalls, iteration_f1s

    @TimeMetric("full_graph_baseline")
    def _evaluate_temporal_full_graph_baseline(
        self,
        run,
        original_graph,
        *,
        dim: int,
        default_p: float,
        default_q: float,
        long_run: bool,
        epochs: int,
    ):
        resolved_p, resolved_q = self._resolved_default_pq(default_p, default_q)
        n_walks, walk_size, window_size = self._walk_params(long_run)
        if "full_model" in run:
            node2vec = run["full_model"]
        else:
            node2vec = Node2Vec(
                dim=dim,
                p=resolved_p,
                q=resolved_q,
                n_walks=n_walks,
                walk_size=walk_size,
                window_size=window_size,
                epochs=epochs,
            )
            node2vec.fit(original_graph, original_graph.nodes)
            run["full_model"] = node2vec
        full_emb = node2vec.embed_nodes(original_graph.nodes)
        full_g = reconstruct(original_graph.number_of_edges(), full_emb, list(original_graph.nodes))
        _, _, full_f1_score = get_f1_score(original_graph, full_g)
        log("F1 score for full graph using Node2Vec: ", full_f1_score)

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
             long_run: bool = False,
             use_dataset_params: bool = False,
             use_lpa: bool = False,
             algorithm: str = "node2vec",
             link_prediction: bool = False,
             checkpoint: str = ""):
        import networkx as nx

        TimeMetric.reset()
        _overall = TimeMetric("test").start()

        run = PersistedRun(checkpoint, name=name, partitions=partitions, alpha=alpha, threshold=threshold, algorithm=algorithm, dim=dim, default_p=default_p, default_q=default_q, epochs=epochs, long_run=long_run, use_dataset_params=use_dataset_params, use_lpa=use_lpa, link_prediction=link_prediction)
        log("Processing dataset ")
        t = VertexEnumerator()
        with TimeMetric("init_dataset"):
            dataset, removed_edges, positive_edges, negative_edges, test_edges = self._initialize_test_dataset(
                run,
                name,
                link_prediction,
                t,
            )
        log(f"Removed {len(removed_edges)} edges for testing link prediction.")
        notify_plugins("test_started", run)
        with TimeMetric("partition"):
            parts = self._partition_for_test(run, dataset, partitions, alpha, threshold, use_lpa)
        notify_plugins("test_partitioned", run)
        log("Total number of nodes: ", dataset.number_of_nodes())
        log("Graph partitioned")
        embs = {}

        for part in parts:
            part_name = hash_set_persistently(part)
            log("Partition size: %d" % len(part))
            if len(part) == 0:
                print("Skipping empty partition")
                continue
            pg = dataset.subgraph(part)
            self._log_partition_graph_stats(nx, pg)
            alg = ALGS[algorithm]

            was_cached, best, best_f1, best_model = self._load_cached_partition_embedding(
                run,
                part_name,
                pg,
                part,
            )
            if was_cached:
                part = list(part)
                if best is None:
                    continue
                for node, e in zip(part, best):
                    self._append_embedding(embs, node, e)
                continue

            with TimeMetric("train_partition"):
              best, best_f1, best_model = self._train_best_partition_model(
                nx=nx,
                alg=alg,
                name=name,
                pg=pg,
                part=part,
                dataset=dataset,
                default_p=default_p,
                default_q=default_q,
                long_run=long_run,
                use_dataset_params=use_dataset_params,
                epochs=epochs,
                dim=dim,
                break_early=break_early,
            )
            log("Best achieved F1 score: ", best_f1)
            notify_plugins("test_partitioned_model_trained", run)
            if best_model is not None:
                run[f"model_{part_name}"] = best_model
            part = list(part)
            if best is None:
                continue
            for node, e in zip(part, best):
                self._append_embedding(embs, node, e)

        if skip_global:
            log("Skipping global F1 computation")
            _overall.stop()
            self._report_timing(run)
            return
        for n in dataset.nodes:
            embs[n] = np.mean(embs[n], axis=0)
        embs = [embs[n] for n in dataset.nodes]
        embedding_dict = {n: embs[i] for i, n in enumerate(dataset.nodes)}
        run["embedding_dict"] = embedding_dict
        if link_prediction:
            self._run_link_prediction_with_embedding(
                run,
                dataset,
                embedding_dict,
                positive_edges,
                negative_edges,
            )

        with TimeMetric("global_f1"):
            g = reconstruct(dataset.number_of_edges(), embs, list(dataset.nodes))
            G = nx.Graph()
            G.add_edges_from(dataset.edges)
            global_precision, global_recall, global_f1 = get_f1_score(G, g)
        log("Global scores: Precision: %f, Recall: %f, F1 Score: %f" % (global_precision, global_recall, global_f1))
        notify_plugins("test_completed", run)
        _overall.stop()
        self._report_timing(run)

    def temporal_test(self, *,
             name: str = "CITESEER",
             partitions: int = 1,
             partitioner_name: str = "random",
             dim: int = 100,
             default_p: float = 0,
             default_q: float = 0,
             epochs: int = 10,
             long_run: bool = False,
             use_dataset_params: bool = False,
             algorithm: str = "dynnode2vec",
             track_seen: bool = False,
             iterations: int = 1,
             limit: int = -1,
             buffer_size: int = 100,
             checkpoint: str = "",
             replication_factor: int = 1,
             checkpoint_iterations: bool = False,
             mu: float = 0,
             epsilon: float = 0.1,
             alpha: float = 1.0,
             skip_full_graph_baseline: bool = False,
               decay: float = 0,
               semantic_metric: str = "cosine",
               semantic_assignment: str = "knn",
             semantic_k: int = -1,
               semantic_eps: float = 0.5,
               semantic_min_samples: int = 2,
               semantic_reassign_noise: bool = True,
               semantic_embedding: str = "auto"):
        import networkx as nx

        TimeMetric.reset()
        _overall = TimeMetric("temporal_test").start()

        scores = []
        log(f"Starting temporal test for dataset {name} with {partitions} partitions and partitioner {partitioner_name} which is embedded in the algorithm {algorithm}.")
        run = PersistedRun(checkpoint, name=name, partitions=partitions, partitioner_name=partitioner_name, dim=dim, default_p=default_p, default_q=default_q, epochs=epochs, long_run=long_run, use_dataset_params=use_dataset_params, algorithm=algorithm, track_seen=track_seen, iterations=iterations, limit=limit, buffer_size=buffer_size, replication_factor=replication_factor, mu=mu, epsilon=epsilon, alpha=alpha, decay=decay, semantic_metric=semantic_metric, semantic_assignment=semantic_assignment, semantic_k=semantic_k, semantic_eps=semantic_eps, semantic_min_samples=semantic_min_samples, semantic_reassign_noise=semantic_reassign_noise, semantic_embedding=semantic_embedding)
        log(f"Processing dataset {name}")
        with TimeMetric("load_dataset"):
            t = VertexEnumerator()
            dataset = _enumerated_event_stream(name, t)
            if limit > 0:
                og_events = list(dataset)[:limit]
            else:
                og_events = list(dataset)
            original_graph = run("graph", to_nx_graph, og_events)
        notify_plugins("temporal_test_started", run)

        logger.debug(f"Original graph has {original_graph.number_of_nodes()} nodes and {original_graph.number_of_edges()} edges")
        if use_dataset_params:
            params: dict = dataset_params.get(name, {})
            dim = params.get('dim', dim)
            default_p = params.get('p', default_p)
            default_q = params.get('q', default_q)

        semantic_n_walks, semantic_walk_size, semantic_window_size = self._walk_params(long_run)
        semantic_resolved_p, semantic_resolved_q = self._resolved_default_pq(default_p, default_q)
        semantic_retrain_threshold = int(0.1 * original_graph.number_of_nodes())
        effective_semantic_k = replication_factor if semantic_k <= 0 else semantic_k
        if semantic_embedding == "auto":
            if partitioner_name == "semantic.node2vec":
                resolved_partitioner_name = "semantic.node2vec"
            elif partitioner_name == "semantic.dynnode2vec":
                resolved_partitioner_name = "semantic.dynnode2vec"
            else:
                resolved_partitioner_name = partitioner_name
        elif semantic_embedding in {"node2vec", "dynnode2vec"}:
            if partitioner_name in {"semantic", "semantic.node2vec", "semantic.dynnode2vec"}:
                resolved_partitioner_name = f"semantic.{semantic_embedding}"
            else:
                resolved_partitioner_name = partitioner_name
        else:
            raise ValueError("semantic_embedding must be one of: auto, node2vec, dynnode2vec")
        for it in range(iterations):
            log(f"Iteration {it+1} / {iterations}: Processing dataset {name}")
            if ("models_%d" % it in run and "partitioner_%d" % it in run
                and "iteration_precisions_%d" % it in run
                and "iteration_recalls_%d" % it in run
                and "iteration_f1s_%d" % it in run):
                log("Loading models and partitioner for iteration...")
                models = run["models_%d" % it]
                partitioner: PartitionerProfile = run["partitioner_%d" % it]
                log("Models and partitioner loaded")
                partitioner.print_profile()
                scores.append(run["iteration_f1s_%d" % it][-1])
            else:
                models = self._create_temporal_models(
                    algorithm=algorithm,
                    partitions=partitions,
                    dim=dim,
                    epochs=epochs,
                    default_p=default_p,
                    default_q=default_q,
                    long_run=long_run,
                    original_graph=original_graph,
                )
                parts: set[Partition] = set(models.keys())
                partitioner = self._create_temporal_partitioner(
                    partitioner_name=resolved_partitioner_name,
                    parts=parts,
                    replication_factor=replication_factor,
                    mu=mu,
                    epsilon=epsilon,
                    alpha=alpha,
                    decay=decay,
                    semantic_metric=semantic_metric,
                    semantic_assignment=semantic_assignment,
                    semantic_k=effective_semantic_k,
                    semantic_eps=semantic_eps,
                    semantic_min_samples=semantic_min_samples,
                    semantic_reassign_noise=semantic_reassign_noise,
                    semantic_dim=dim,
                    semantic_epochs=epochs,
                    semantic_p=semantic_resolved_p,
                    semantic_q=semantic_resolved_q,
                    semantic_n_walks=semantic_n_walks,
                    semantic_walk_size=semantic_walk_size,
                    semantic_window_size=semantic_window_size,
                    semantic_retrain_threshold=semantic_retrain_threshold,
                )
                sorted_events = self._sort_temporal_events(og_events, track_seen)
                old_f1_score, iteration_precisions, iteration_recalls, iteration_f1s = self._process_temporal_buffers(
                    nx=nx,
                    sorted_events=sorted_events,
                    buffer_size=buffer_size,
                    partitioner=partitioner,
                    models=models,
                    original_graph=original_graph,
                    iteration=it,
                    run = run if checkpoint_iterations else None
                )
                log("Event stream processing completed")
                scores.append(old_f1_score)
                run["models_%d" % it] = models
                run["partitioner_%d" % it] = partitioner
                run["iteration_precisions_%d" % it] = iteration_precisions
                run["iteration_recalls_%d" % it] = iteration_recalls
                run["iteration_f1s_%d" % it] = iteration_f1s
                partitioner.print_profile()
                run["profile_%d" % it] = partitioner.dump_profile()
        log("Average F1 score: ", np.mean(scores))
        log("Standard deviation of F1 score: ", np.std(scores))
        if not skip_full_graph_baseline:
            log("Evaluating full graph baseline...")
            self._evaluate_temporal_full_graph_baseline(
                run,
                original_graph,
                dim=dim,
                default_p=default_p,
                default_q=default_q,
                long_run=long_run,
                epochs=epochs,
            )
        notify_plugins("temporal_test_completed", run)
        _overall.stop()
        self._report_timing(run)