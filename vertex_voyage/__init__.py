
import networkx as nx  
import cdlib.algorithms
import cdlib.utils
import numpy as np
import vertex_voyage.node2vec as nv 
from vertex_voyage.cluster import *
import os 
from vertex_voyage.partitioning import partition_graph
from vertex_voyage.node2vec import Node2Vec

class StorageGraph:

    def __init__(self, name: str):
        self.name = name
        self.GRAPH_STORE_PATH = os.environ.get("GRAPH_STORE_PATH", os.environ.get("HOME") + "/.vertex_voyage/graphs")
        if not os.path.exists(self.GRAPH_STORE_PATH):
            os.makedirs(self.GRAPH_STORE_PATH)
        self.path = os.path.join(self.GRAPH_STORE_PATH, name + ".gml")
    
    def get_partition(self, index: int) -> "StorageGraph":
        return StorageGraph(self.name + "_part_" + str(index))

    def create_graph(self, graph: nx.Graph):
        nx.write_gml(graph, self.path)
        return self.path

    def get_graph(self) -> nx.Graph:
        return nx.read_gml(self.path)
    
    def add_vertex(self, vertex_name):
        graph = self.get_graph()
        graph.add_node(vertex_name)
        self.create_graph(graph)
        return self.path

    def add_edge(self, vertex1, vertex2):
        graph = self.get_graph()
        graph.add_edge(vertex1, vertex2)
        self.create_graph(graph)
        return self.path
    
    def partition_graph(self, num_nodes: int) -> list:
        graph = self.get_graph()
        partitioned_graph = partition_graph(graph, num_nodes)
        result = [] 
        for i, part in enumerate(partitioned_graph):
            storage_graph = self.get_partition(i)
            storage_graph.create_graph(part)
            result.append(storage_graph)
        return result
    
    def get_node_num(self):
        """
        Gets node number from GML file. This does not load whole graph into memory.
        """
        counter = 0
        with open(self.path, "r") as f:
            for line in f:
                if "node" in line:
                    counter += 1
        return counter
    
    def get_nodes(self):
        """
        Gets list of nodes from GML file without loading whole graph into memory.
        """
        result = []
        with open(self.path, "r") as f:
            for line in f:
                if "node" in line:
                    result.append(line.split()[1])
        return result 



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
            return StorageGraph(graph_name).partition_graph(self.get_node_number())
        else:
            do_rpc_to_leader("partition_graph", graph_name=graph_name)
    def get_embedding(self, graph_name: str):
        if is_leader():
            current_node_index = get_node_index(get_current_node())
            partitioned_graph = StorageGraph(graph_name).get_partition(current_node_index).get_graph()
            nv = Node2Vec(
                dim=128, 
                epochs=10,
                learning_rate=0.01,
                n_walks=10,
                negative_sample_num=1,
                p=1,
                q=1,
                window_size=10,
                walk_size=10
            )
            nodes = StorageGraph(graph_name).get_nodes()
            nv.fit(partitioned_graph, nodes)
            nodes_on_current_node = list(partitioned_graph.nodes())
            embeddings = nv.embed_nodes(nodes_on_current_node)
            return embeddings
        else:
            do_rpc_to_leader("get_embedding", graph_name=graph_name)    
    
    def get_vertices(self, graph_name: str):
        if is_leader():
            return StorageGraph(graph_name).get_nodes()
        else:
            do_rpc_to_leader("get_vertices", graph_name=graph_name)
    def kmeans(self, graph_name: str):
        return None
    def walk(self, graph_name: str):
        return None
    def cluster_similarity(self, graph_name1: str, graph_name2: str):
        return None
    def graph_similarity(self, graph_name1: str, graph_name2: str):
        return None
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
    
    def process(self, graph_name: str):
        if get_leader() == get_current_node():
            my_embedding = self.get_embedding(graph_name)
            nodes = get_nodes()
            # do xmlrpc to other nodes and add their embeddings to my_embedding
            for node in nodes:
                if node == get_current_node():
                    continue
                embedding = do_rpc(get_node_index(node), "get_embedding", graph_name)
                my_embedding = np.concatenate((my_embedding, embedding))
            return my_embedding
        else:
            do_rpc_to_leader("process", graph_name)

COMMAND_CLASSES = ["Executor"]