
from vertex_voyage.cluster import * 
from vertex_voyage.command_executor import *
import termcolor
import time 

def red(text):
    return termcolor.colored(text, "red")

def green(text):
    return termcolor.colored(text, "green")

def yellow(text):
    return termcolor.colored(text, "yellow")

def print_step(text):
    print("* " + green(text))

def print_substep(text):
    print("  - " + yellow(text))

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
    def process(self, graph_name: str, *, ip: str="localhost", 
                dim: int = 128,
                epochs: int=10,
                learning_rate: float=0.01,
                n_walks: int=10,
                negative_sample_num: int=1,
                p: float=1,
                q: float=1,
                window_size: int=10,
                walk_size: int=10
        ):
        return do_rpc_client(ip, "process", 
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

    def get_edges(self, graph_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_edges", graph_name=graph_name)

    def get_datasets(self, datasets: str, *, ip: str = "localhost", no_sbm: bool = False):
        datasets_file = open(datasets, "r")
        sbms = [
            {
                "size": 10000,
                "p": 0.1,
                "q": 0.01,
                "communities": 2
            }
        ]
        if no_sbm:
            sbms = []
        for i, sbm in enumerate(sbms):
            do_rpc_client(ip, "generate_graph", 
                graph_name=f"sbm{i}",
                n=sbm["size"],
                p=sbm["p"],
                q=sbm["q"],
                c=sbm["communities"]
            )
        for line in datasets_file:
            line = line.strip()
            if line == "":
                continue
            if line.startswith("#"):
                continue
            path = line 
            name = path.split("/")[-1].split(".")[0]
            if path.endswith(".gml"):
                self.upload_gml(name, path, ip=ip)
            else:
                self.upload_csv(name, path, ip=ip)
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
                data = f.read(5*1024*1024)
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
    
    def execute(self, pipeline: str, *, ip: str = "localhost"):
        """
        Execute commands specified in pipeline YAML file given. 

        Pipeline file contains name of pipeline and list of commands to execute in order.
        Every command has its name which corresponds to a method in the Client class and its arguments.

        Result of each execution is saved into file inside folder which is named after the pipeline name.

        Result is JSON file which is named after the order of the command in the pipeline.

        Command in pipeline can be of several types:

        - oneshot - execute exactly as specified
        - vary - execute with different values of parameter
            - parameter - name of parameter to vary
            - start - start value of parameter
            - end - end value of parameter
            - step - step value of parameter
        - multiple - execute with multiple values of parameter
            - parameter - name of parameter to vary
            - values - list of values to use

        Args:
            pipeline (str): Path to pipeline YAML file
            ip (str, optional): IP address of server. Defaults to "localhost".
        """
        import yaml 
        import json
        with open(pipeline, "r") as f:
            pipeline = yaml.safe_load(f)
        pipeline_name = pipeline["name"]
        print_step(f"Executing pipeline {pipeline_name}")
        if not os.path.exists(pipeline_name):
            os.makedirs(pipeline_name)
        with open(f"{pipeline_name}/pipeline.yaml", "w") as f:
            yaml.safe_dump(pipeline, f)
        for i, command in enumerate(pipeline["commands"]):
            command_name = command["name"]
            print_substep(f"Executing command {command_name}")
            command_result_name = f"{pipeline_name}/{i}_{command_name}.json"
            if command_name not in dir(self):
                print(f"Command {command_name} not found in Client class")
                return 
            if command["type"] == "oneshot":
                start_time = time.time()
                result = getattr(self, command_name)(**command["args"], ip=ip)
                end_time = time.time()
                with open(command_result_name, "w") as f:
                    json.dump({
                        "result": result,
                        "time": (end_time - start_time)
                    }, f)
            elif command["type"] == "vary":
                parameter = command["parameter"]
                start = command["start"]
                end = command["end"]
                step = command["step"]
                values = list(range(start, end, step))
                results = []
                for value in values:
                    print(f"Varying {parameter} to {value} / {end}" + 16*" ", end="\r")
                    start_time = time.time()
                    result = getattr(self, command_name)(**{**command["args"], parameter: value}, ip=ip)
                    end_time = time.time()
                    results.append({
                        parameter: value,
                        "result": result,
                        "time": (end_time - start_time)
                    })
                with open(command_result_name, "w") as f:
                    json.dump(results, f)
            elif command["type"] == "multiple":
                parameter = command["parameter"]
                values = command["values"]
                results = []
                for value in values:
                    print(f"Varying {parameter} to {value}" + 16*" ", end="\r")
                    start_time = time.time()
                    result = getattr(self, command_name)(**{**command["args"], parameter: value}, ip=ip)
                    end_time = time.time()
                    results.append({
                        parameter: value,
                        "result": result,
                        "time": (end_time - start_time)
                    })
                with open(command_result_name, "w") as f:
                    json.dump(results, f)
            else:
                print(f"Unknown command type {command['type']}")
        return {
            "pipeline": pipeline_name,
            "Results folder": os.path.abspath(pipeline_name)
        }

    def analyze_embeddings(self, single_node_result: str, multi_node_result: str, clusters: int):
        import json 
        with open(single_node_result, "r") as f:
            single_node_results = json.load(f)
        with open(multi_node_result, "r") as f:
            multi_node_results = json.load(f)
        x_label = [k for k in single_node_results[0].keys() if k != "result" and k != "time"][0]
        y_label = "Cluster similarity"
        import pandas as pd 
        x = []
        y = []
        for single_node_result, multi_node_result in zip(single_node_results, multi_node_results):
            x.append(single_node_result[x_label])
            keys = sorted(single_node_result["result"]["embeddings"].keys())
            single = single_node_result["result"]["embeddings"]
            multi = multi_node_result["result"]["embeddings"]
            single = [single[key] for key in keys]
            multi = [multi[key] for key in keys]
            from sklearn.cluster import KMeans
            single_k_means = KMeans(n_clusters=clusters).fit_predict(single)
            multi_k_means = KMeans(n_clusters=clusters).fit_predict(multi)
            from sklearn.metrics import adjusted_rand_score
            similarity = adjusted_rand_score(single_k_means, multi_k_means)
            y.append(similarity)
        df = pd.DataFrame({
            x_label: x,
            y_label: y
        })
        df.to_csv("analysis.csv")

    def analyze_reconstruction(self, vertex_result: str, edge_result: str, embeddings_result: str):
        import json 
        import pandas as pd
        from vertex_voyage.reconstruction import reconstruct, get_f1_score
        import networkx as nx 
        import numpy as np
        vertex_result = json.load(open(vertex_result, "r"))
        edge_result = json.load(open(edge_result, "r"))
        embeddings_result = json.load(open(embeddings_result, "r"))
        x_label = [k for k in embeddings_result[0].keys() if k != "result" and k != "time"][0]
        y_label = "F1 score"
        x = []
        y = []
        for embedding in embeddings_result:
            x.append(embedding[x_label])
            vertices = vertex_result["result"]
            edges = edge_result["result"]
            original = nx.Graph()
            emb = embedding["result"]["embeddings"]
            emb = {int(k): v for k, v in emb.items()}
            keys = sorted(emb.keys())
            emb = [emb[key] for key in keys]
            emb = [np.array(e) for e in emb]
            print(emb)
            for vertex in vertices:
                original.add_node(vertex)
            for edge in edges:
                original.add_edge(edge[0], edge[1])
            reconstructed = reconstruct(len(edges), emb, vertices)
            y.append(get_f1_score(original, reconstructed))
        df = pd.DataFrame({
            x_label: x,
            y_label: y
        })
        df.to_csv("analysis.csv")
        


        


COMMAND_CLASSES = ["Client"]