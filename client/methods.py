
from vertex_voyage.cluster import * 
from vertex_voyage.command_executor import *

class Client:
    def create_graph(self, name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "create_graph", name=name)
    def add_vertex(self, graph_name: str, vertex_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "add_vertex", graph_name=graph_name, vertex_name=vertex_name)
    def add_edge(self, graph_name: str, vertex1: str, vertex2: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "add_edge", graph_name=graph_name, vertex1=vertex1, vertex2=vertex2)
    def partition_graph(self, graph_name: str, num_nodes: str, *, ip: str="localhost"):
        return do_rpc_client(ip, "partition_graph", graph_name=graph_name, num_nodes=num_nodes)
    def get_node_num(self, graph_name: str, *, ip: str="localhost"):
        return do_rpc_client(ip, "get_node_num", graph_name=graph_name)
    def get_nodes(self, graph_name: str, *, ip: str="localhost"):
        return do_rpc_client(ip, "get_nodes", graph_name=graph_name)
    def get_partition(self, graph_name: str, partition_num: int, *, ip: str="localhost"):
        return do_rpc_client(ip, "get_partition", graph_name=graph_name, partition_num=partition_num)
    def process(self, graph_name: str, partition_num: int, *, ip: str="localhost", dim: int = 128):
        return do_rpc_client(ip, "process", graph_name=graph_name, partition_num=partition_num, dim=dim)
    def get_graph(self, graph_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_graph", graph_name=graph_name)
    def get_leader(self, *, ip: str = "localhost"):
        print("get_leader called")
        return do_rpc_client(ip, "get_leader")

    def import_karate_club(self, name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "import_karate_club", name=name)
        
COMMAND_CLASSES = ["Client"]
