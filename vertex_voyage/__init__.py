
import networkx as nx  
import cdlib.algorithms
import cdlib.utils
import numpy as np
import vertex_voyage.node2vec as nv 
from vertex_voyage.cluster import *
import os 
from vertex_voyage.partitioning import partition_graph, calculate_corruptability
from vertex_voyage.node2vec import Node2Vec
from httplib2 import Http
import concurrent.futures
import time 
import vertex_voyage.config as cfg 

class ControlInterface:

    additional_classes = [] 
    threads = []

    def add_command_class(self, cls):
        ControlInterface.additional_classes.append(cls)
    
    def start_background_thread(self, target, *args, **kwargs):
        t = threading.Thread(target=target, args=args, kwargs=kwargs)
        t.start()
        ControlInterface.threads.append(t)


def parallel_function_call(func, param_list, max_workers=None):
    """
    Parallelizes the execution of a function with a parameter list.
    
    Args:
        func (callable): The function to be executed in parallel.
        param_list (list): A list of parameters where each parameter is passed to func.
        max_workers (int, optional): The maximum number of threads to use. Defaults to None.
        
    Returns:
        list: A list of results from the function calls.
    """
    results = []
    
    # Use ThreadPoolExecutor or ProcessPoolExecutor based on your needs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the function and the parameter list to the executor
        future_to_param = {executor.submit(func, param): param for param in param_list}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_param):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Function call raised an exception: {e}")
    
    print("All function calls completed.")
    print("Results:", results)
    return results


COMMAND_CLASSES = ["Executor"]
class StorageGraph:
    GRAPH_STORE_PATH = os.environ.get("GRAPH_STORE_PATH", os.environ.get("HOME") + "/.vertex_voyage/graphs")

    def __init__(self, name: str):
        self.name = name
        if not os.path.exists(self.GRAPH_STORE_PATH):
            os.makedirs(self.GRAPH_STORE_PATH)
        self.path = os.path.join(self.GRAPH_STORE_PATH, name + ".gml")
    
    def list():
        if not os.path.exists(StorageGraph.GRAPH_STORE_PATH):
            return []
        return [s.replace(".gml", "") for s in os.listdir(StorageGraph.GRAPH_STORE_PATH)]

    def get_partition(self, index: int) -> "StorageGraph":
        try:
            return StorageGraph(self.name + "_part_" + str(index))
        except FileNotFoundError:
            raise ValueError(f"Partition {index} does not exist.")
        

    def create_graph(self, graph: nx.Graph):
        nx.write_gml(graph, self.path)
        cfg.notify_plugins("graph_created", self.path)
        return self.path

    def get_graph(self) -> nx.Graph:
        G = nx.read_gml(self.path)
        mapping = {v: int(v) for v in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        return G
    
    def add_vertex(self, vertex_name):
        graph = self.get_graph()
        graph.add_node(vertex_name)
        self.create_graph(graph)
        cfg.notify_plugins("vertex_added", self.path, graph, vertex_name)
        return self.path

    def add_edge(self, vertex1, vertex2):
        graph = self.get_graph()
        graph.add_edge(vertex1, vertex2)
        self.create_graph(graph)
        cfg.notify_plugins("edge_added", self.path, graph, vertex1, vertex2)
        return self.path
    
    def partition_graph(self, num_nodes: int) -> list:
        if num_nodes == 1:
            return [self.copy(self.name + "_part_0")], 0
        graph = self.get_graph()
        partitioned_graph = partition_graph(graph, num_nodes)
        corruptability = calculate_corruptability(graph, num_nodes, partitions=partitioned_graph)
        print("Partitioned graph:", partitioned_graph, flush=True)
        result = [] 
        for i, part in enumerate(partitioned_graph):
            print("Creating partition %d with nodes %s" % (i, part), flush=True)
            storage_graph = self.get_partition(i)
            subgraph = graph.subgraph(part)
            storage_graph.create_graph(subgraph)
            result.append(storage_graph)
        cfg.notify_plugins("graph_partitioned", self.path, graph, partitioned_graph, corruptability, result)
        return result, corruptability
    
    def get_node_num(self):
        graph = self.get_graph()
        return len(graph.nodes())
    
    def get_nodes(self):
        graph = self.get_graph()
        return list(graph.nodes())

    def import_from_url(self, url: str):
        h = Http()
        resp, content = h.request(url, "GET")
        with open(self.path, "wb") as f:
            f.write(content)
        return self.path

    def copy(self, destination_name: str):
        import shutil
        shutil.copy(self.path, os.path.join(self.GRAPH_STORE_PATH, destination_name + ".gml"))
        return os.path.join(self.GRAPH_STORE_PATH, destination_name + ".gml")

@cfg.pluggable
def get_graph_storage_class(default_storage_class):
    return default_storage_class

StorageGraph = get_graph_storage_class(StorageGraph)
class Executor:
    def create_graph(self, name: str):
        if is_leader():
            return StorageGraph(name).create_graph(nx.Graph())
        else:
            do_rpc_to_leader("create_graph", name=name)
    def add_vertex(self, graph_name: str, vertex_name: str):
        if is_leader():
            return StorageGraph(graph_name).add_vertex(vertex_name)
        else:
            do_rpc_to_leader("add_vertex", graph_name=graph_name, vertex_name=vertex_name)
    def add_edge(self, graph_name: str, vertex1: str, vertex2: str):
        if is_leader:
            return StorageGraph(graph_name).add_edge(vertex1, vertex2)
        else:
            do_rpc_to_leader("add_edge", graph_name=graph_name, vertex1=vertex1, vertex2=vertex2)
    def partition_graph(self, graph_name: str):
        if is_leader():
            partitions, corruptability = StorageGraph(graph_name).partition_graph(len(get_nodes()))
            return {
                "partition_count": len(get_nodes()),
                "partitions": partitions,
                "corruptability": corruptability
            }
        else:
            return do_rpc_to_leader("partition_graph", graph_name=graph_name)
    
    def get_partition(self, graph_name: str, partition_num: int):
        if is_leader():
            try:
                part = StorageGraph(graph_name).get_partition(partition_num)
                return {
                    "name": part.name,
                    "path": part.path,
                    "nodes": list(part.get_graph().nodes()),
                    "edges": list(part.get_graph().edges())
                }
            except FileNotFoundError:
                return {
                    "error": f"Partition {partition_num} does not exist."
                }
        else:
            return do_rpc_to_leader("get_partition", graph_name=graph_name, partition_num=partition_num)
    def get_embedding(self, 
                      graph_name: str, *, 
                      dim: int = 128,
                        epochs: int = 10,
                        learning_rate: float = 0.01,
                        n_walks: int = 10,
                        negative_sample_num: int = 1,
                        p: float = 1,
                        q: float = 1,
                        window_size: int = 10,
                        walk_size: int = 10
        ):
        print("Getting embedding", flush=True)
        current_node_index = get_node_index(get_current_node())
        if is_leader():
            partitioned_graph = StorageGraph(graph_name).get_partition(current_node_index).get_graph()
        else:
            partitioned_graph = do_rpc_to_leader("get_partition", graph_name=graph_name, partition_num=current_node_index)
            g = nx.Graph()
            for v in partitioned_graph["nodes"]:
                g.add_node(v)
            for e in partitioned_graph["edges"]:
                g.add_edge(*e)
            partitioned_graph = g
        print("Partitioned graph:", partitioned_graph.nodes(), flush=True)
        print("Current node:", get_current_node(), flush=True)
        print("Launching node2vec", flush=True)
        nv = Node2Vec(
            dim=dim, 
            epochs=epochs,
            learning_rate=learning_rate,
            n_walks=n_walks,
            negative_sample_num=negative_sample_num,
            p=p,
            q=q,
            window_size=window_size,
            walk_size=walk_size
        )
        if is_leader():
            nodes = StorageGraph(graph_name).get_nodes()
        else:
            nodes = do_rpc_to_leader("get_vertices", graph_name=graph_name)
        nv.fit(partitioned_graph, nodes)
        nodes_on_current_node = list(partitioned_graph.nodes())
        print("Nodes on current node:", nodes_on_current_node, flush=True)
        embeddings = {node: nv.embed_node(node) if node in nodes_on_current_node else np.zeros(dim) for node in nodes}
        embeddings = {str(node): embeddings[node].tolist() for node in embeddings}
        return embeddings
    def get_vertices(self, graph_name: str):
        if is_leader():
            return StorageGraph(graph_name).get_nodes()
        else:
            do_rpc_to_leader("get_vertices", graph_name=graph_name)

    def get_edges(self, graph_name: str):
        if is_leader():
            return list(StorageGraph(graph_name).get_graph().edges())
        else:
            do_rpc_to_leader("get_edges", graph_name=graph_name)

    def import_karate_club(self, name: str):
        if is_leader():
            graph = nx.karate_club_graph()
            return StorageGraph(name).create_graph(graph)
        else:
            do_rpc_to_leader("import_karate_club", name=name)
    def get_leader(self):
        return get_leader()
        

    def zk(self):
        client = get_zk_client()
        register_node()
        nodes = get_nodes()
        node_data = get_node_data(nodes[0])
        node_index = get_node_index(nodes[0])
        node_by_index = get_node_by_index(1)
        leader = get_leader()
        current_node = get_current_node()
        return {
            "node_data": node_data,
            "node_index": node_index,
            "node_by_index": node_by_index,
            "leader": leader,
            "current_node": current_node
        }
    
    def process(self, 
                graph_name: str, *, 
                dim: int=128,
                epochs: int=10,
                learning_rate: float=0.01,
                n_walks: int=10,
                negative_sample_num: int=1,
                p: float=1,
                q: float=1,
                window_size: int=10,
                walk_size: int=10
        ):
        if is_leader():
            print("Processing on leader", flush=True)
            start_time = time.time()
            end_time = None 
            my_embedding = self.get_embedding(
                graph_name, 
                dim=dim,
                epochs=epochs,
                walk_size=walk_size,
                n_walks=n_walks,
                window_size=window_size,
                negative_sample_num=negative_sample_num,
                p=p,
                q=q,
                learning_rate=learning_rate
            )
            my_embedding = {int(k): np.array(v) for k, v in my_embedding.items()}
            nodes = get_nodes()
            graph_vertices = StorageGraph(graph_name).get_nodes()
            print("Nodes:", nodes, flush=True)
            node_to_embeddings_count = {n: 0 for n in graph_vertices}
            for k in my_embedding:
                node_to_embeddings_count[k] += 1
            # do xmlrpc to other nodes and add their embeddings to my_embedding
            if len(nodes) > 1:
                def f_exec(node):
                    print("Getting embedding from", node, flush=True)
                    return do_rpc(
                        get_node_index(node), 
                        "get_embedding", 
                        graph_name=graph_name, 
                        dim=dim,
                        epochs=epochs,
                        walk_size=walk_size,
                        n_walks=n_walks,
                        window_size=window_size,
                        negative_sample_num=negative_sample_num,
                        p=p,
                        q=q,
                        learning_rate=learning_rate
                    )
                other_nodes = [n for n in nodes if n != get_current_node()]
                print("Other nodes:", other_nodes, flush=True)
                embeddings = parallel_function_call(f_exec, other_nodes)
                end_time = time.time()
                for embedding in embeddings:
                    print("Embedding from other node:", embedding, flush=True)
                    for k, v in embedding.items():
                        k = int(k)
                        print("Adding embedding for", k, flush=True)
                        node_to_embeddings_count[k] += 1
                        if k not in my_embedding:
                            my_embedding[k] = v
                        else:
                            my_embedding[k] = np.array(my_embedding[k]) + np.array(v)
            else:
                end_time = time.time()
            for k, v in my_embedding.items():
                my_embedding[k] = np.array(v)
            print("Normalizing embeddings", flush=True)
            for k in my_embedding:
                my_embedding[k] = my_embedding[k] / node_to_embeddings_count[k]
            return {
                "embeddings": {str(k): my_embedding[k].tolist() for k in my_embedding},
                "time": end_time - start_time
            }
        else:
            return do_rpc_to_leader("process", graph_name)

    def import_gml(self, url: str, graph_name: str):
        """
        Imports a GML file from a URL.

        Parameters:
        - url (str): URL of the GML file.
        - graph_name (str): Name of the graph to import.

        Returns:
        - str: Path to the imported GML file.
        """
        if is_leader():
            print("Importing from URL", url, flush=True)
            return StorageGraph(graph_name).import_from_url(url)
        else:
            do_rpc_to_leader("import_gml", url=url, graph_name=graph_name)


    def download_gml(self, dataset_name: str, *, source: str="snap"):
        """
        Downloads a GML file for the given dataset name from the specified source.

        Parameters:
        - dataset_name (str): Name of the dataset/graph to download.
        - source (str): The source to download from. Options are "snap", "konect", or "network_repository".
        - save_dir (str): Directory to save the downloaded GML file.

        Returns:
        - str: Path to the downloaded GML file.
        """
        base_urls = {
            "snap": f"https://snap.stanford.edu/data/{dataset_name}.gml",
            "konect": f"http://konect.cc/files/download.tsv.{dataset_name}.gml",
            "network_repository": f"http://networkrepository.com/{dataset_name}.gml"
        }
        if not is_leader():
            return do_rpc_to_leader("download_gml", dataset_name=dataset_name, source=source)
        if source not in base_urls:
            raise ValueError(f"Source '{source}' is not supported. Use 'snap', 'konect', or 'network_repository'.")
        print("Downloading GML", dataset_name, source, flush=True)
        url = base_urls[source]
        return self.import_gml(url, dataset_name)
    
    def generate_graph(self, graph_name: str, n: int, p: float, q: float, c: int):
        """
        Generates a planted partition graph.

        Parameters:
        - graph_name (str): Name of the graph.
        - n (int): Number of vertices.
        - p (float): Probability of intra-community edges.
        - q (float): Probability of inter-community edges.
        - c (int): Number of communities.

        Returns:
        - str: Path to the generated GML file.
        """
        if is_leader():
            graph = nx.planted_partition_graph(c, n, p, q)
            print("Vertex count: ", len(graph.nodes()), flush=True)
            print("Edge count: ", len(graph.edges()), flush=True)
            g = nx.Graph()
            for v in graph.nodes():
                g.add_node(v)
            for e in graph.edges():
                g.add_edge(*e)
            return StorageGraph(graph_name).create_graph(g)
        else:
            do_rpc_to_leader("generate_graph", graph_name=graph_name, sizes=sizes, p_matrix=p_matrix)
    def list(self):
        if is_leader():
            return StorageGraph.list()
        else:
            return do_rpc_to_leader("list")
    
    def upload_gml(self, graph_name: str, data: bytes, *, append: bool=False):
        print("Uploading GML", flush=True)
        if is_leader():
            try:
                path = StorageGraph(graph_name).path
                print("Path:", path, flush=True)
                size = 0
                print("Uploading GML chunk of size ", len(data), flush=True)
                with open(path, "ab" if append else "wb") as f:
                    size = f.write(data)
                return size
            except: 
                return -1
        else:
            return do_rpc_to_leader("upload_gml", graph_name=graph_name, data=data, append=append)
    
