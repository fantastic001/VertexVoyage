
from vertex_voyage.cluster import * 
from vertex_voyage.command_executor import *

class Client:
    def create_graph(self, name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "create_graph", name=name)
    def add_vertex(self, graph_name: str, vertex_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "add_vertex", graph_name=graph_name, vertex_name=vertex_name)
    def add_edge(self, graph_name: str, vertex1: str, vertex2: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "add_edge", graph_name=graph_name, vertex1=vertex1, vertex2=vertex2)
    def partition_graph(self, graph_name: str, *, ip: str="localhost"):
        return do_rpc_client(ip, "partition_graph", graph_name=graph_name)
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
    def get_embedding(self, graph_name: str, *, ip: str = "localhost", dim: int = 128):
        return do_rpc_client(ip, "get_embedding", graph_name=graph_name, dim=dim)

    def import_karate_club(self, name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "import_karate_club", name=name)
    
    def get_vertices(self, graph_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_vertices", graph_name=graph_name)

    def get_datasets(self, *, ip: str = "localhost", no_sbm: bool = False):
        snap_datasets = [
            "twitch-gamers"
        ]
        sbms = [
            {
                "sizes": [100000, 100000],
                "p_matrix": [[0.1, 0.01], [0.01, 0.1]],
            },
            {
                "sizes": [100000, 100000],
                "p_matrix": [[0.5, 0.01], [0.01, 0.5]],
            }
        ]
        if no_sbm:
            sbms = []
        for i, sbm in enumerate(sbms):
            do_rpc_client(ip, "generate_graph", 
                graph_name=f"sbm{i}",
                sizes=sbm["sizes"],
                p_matrix=sbm["p_matrix"]
            )
            dataset_counter += 1
        for i, dataset in enumerate(snap_datasets):
            do_rpc_client(ip, "download_gml", 
                dataset_name=dataset,
                graph_name=f"snap{i}"
            )
        datasets = do_rpc_client(ip, "list")
        return datasets

    def upload_gml(self, graph_name: str, path: str, *, ip: str = "localhost"):
        print("Uploading GML")
        size = do_rpc_client(ip, "upload_gml", graph_name=graph_name, data=b"")
        if not isinstance(size, int):
            return size
        if size < 0:
            return "Error uploading graph"
        # read in chunks of 1024 bytes
        file_size = os.path.getsize(path)
        print(f"File size: {file_size}")
        read = 0
        with open(path, "rb") as f:
            while True:
                data = f.read(1024)
                read += len(data)
                print(f"Read {read}/{file_size} bytes", end="\r")
                if not data:
                    break
                result = do_rpc_client(ip, "upload_gml", graph_name=graph_name, data=data, append=True)
                size = 0 
                if isinstance(result, int):
                    size = result
                else:
                    return result
                if size < 0:
                    return "Error uploading graph"
    def upload_csv(self, graph_name: str, path: str, *, ip: str = "localhost", limit: int = None):
        # create graph of edges in CSV 
        import networkx as nx 
        g = nx.Graph()
        with open(path, "r") as f:
            if limit is not None:
                lines = f.readlines()[:limit]
            else:
                lines = f.readlines()
            for line in lines[1:]:
                u, v = line.strip().split(",")
                g.add_edge(u, v)
        tmpfile = f"/tmp/{graph_name}.gml"
        print(f"Writing to {tmpfile}")
        nx.write_gml(g, tmpfile)
        print("Uploading GML")
        return self.upload_gml(graph_name, tmpfile, ip=ip)
    
    def list(self, *, ip: str = "localhost"):
        return do_rpc_client(ip, "list")



COMMAND_CLASSES = ["Client"]