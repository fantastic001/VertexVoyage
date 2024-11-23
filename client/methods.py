
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
    
    def import_known_graph(self, name: str, graph_name: str, *, ip: str = "localhost"):
        import networkx as nx 
        G = getattr(nx, name)()
        vv_dir = os.path.join("/tmp", "vertex_voyage")
        if not os.path.exists(vv_dir):
            os.makedirs(vv_dir)
        tmpfile = os.path.join("/tmp", "vertex_voyage", f"{name}.gml")
        # replace vertex names with numbers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        nx.write_gml(G, tmpfile)
        return self.upload_gml(graph_name, tmpfile, ip=ip)
    
    def get_vertices(self, graph_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_vertices", graph_name=graph_name)

    def get_edges(self, graph_name: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_edges", graph_name=graph_name)

    def get_datasets(self, datasets: str, *, ip: str = "localhost", no_sbm: bool = False):
        datasets_file = open(datasets, "r")
        sbms = [
            {
                "size": 1000,
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
    def upload(self, graph_name: str, path: str, *, ip: str = "localhost", sep: str = "\t", skiplines: int = 0, limit: int = None, jsonpath: str = None):
        if path.endswith(".gml"):
            return self.upload_gml(graph_name, path, ip=ip)
        tmpfile = os.path.join("/tmp", "vertex_voyage", f"{graph_name}.gml")
        return self.upload_gml(graph_name, self.convert_to_gml(path, tmpfile, sep=sep, skiplines=skiplines, limit=limit, jsonpath=jsonpath), ip=ip)
    
    def convert_to_gml(self, path: str, output: str, *, sep: str = "\t", jsonpath: str = None, skiplines: int = 0, limit: int = None):
        import networkx as nx 
        g = nx.Graph()
        if path.endswith(".gml"):
            return path
        if path.endswith(".json"):
            import json 
            with open(path, "r") as f:
                data = f.read()
                data = json.loads(data)
                if jsonpath is not None:
                    for key in jsonpath.split("."):
                        data = data[key]
                for edge in data:
                    g.add_edge(edge[0], edge[1])
        elif path.endswith(".tsv"):
            with open(path, "r") as f:
                for line in f.readlines()[skiplines:limit]:
                    u, v = line.strip().split(sep)
                    g.add_edge(u, v)
        elif path.endswith(".csv"):
            with open(path, "r") as f:
                for line in f.readlines()[skiplines:limit]:
                    u, v = line.strip().split(",")
                    g.add_edge(u, v)
        else:
            return "Unsupported format"
        print("Writing to", output)
        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))
        nx.write_gml(g, output)
        return output
    
    def list(self, *, ip: str = "localhost"):
        return do_rpc_client(ip, "list")
    
    def get_config_location(self, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_config_location")
    
    def get_config(self, key: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "get_config", key=key)
    
    def set_config(self, key: str, value: str, *, ip: str = "localhost"):
        return do_rpc_client(ip, "set_config", key=key, value=value)
    
    def execute(self, pipeline: str, *, ip: str = "localhost", results_folder: str = None):
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
        if results_folder is None:
            results_folder = pipeline_name
        print_step(f"Executing pipeline {pipeline_name}")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        with open(f"{results_folder}/pipeline.yaml", "w") as f:
            yaml.safe_dump(pipeline, f)
        for i, command in enumerate(pipeline["commands"]):
            command_name = command["name"]
            print_substep(f"Executing command {command_name}")
            command_result_name = f"{results_folder}/{i}_{command_name}.json"
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
            "Results folder": os.path.abspath(results_folder)
        }

    def analyze_embeddings(self, single_node_result: str, multi_node_result: str, clusters: int, output: str):
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
        # create parent directory if nonexisting 
        parent_dir = os.path.dirname(output)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        df.to_csv(output)

    def analyze_reconstruction(self, vertex_result: str, edge_result: str, embeddings_result: str, output: str):
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
            try:
                y.append(get_f1_score(original, reconstructed))
            except ZeroDivisionError:
                y.append(0)
        df = pd.DataFrame({
            x_label: x,
            y_label: y
        })
        print("Saving to", output)
        df.to_csv(output)
    
    def analyze_sbm_corruptability(self):
        import pandas as pd 
        from vertex_voyage.partitioning import calculate_graph_corruptability
        import networkx as nx 
        import numpy as np 
        import matplotlib.pyplot as plt
        qs = list(np.linspace(0.00, 0.03, 30))
        ps = [0.4, 0.5, 0.6]
        plt.title("Koruptabilnost SBM modela u zavisnosti od verovatnoće veza između zajednica")
        plt.xlabel("q")
        plt.ylabel("Koruptabilnost")
        for p in ps:
            corruptabilities = []
            print(f"Calculating for p={p}")
            for q in qs:
                print(f"Calculating for q={q}     ", end="")
                # create SBM with community relation probability 0.1 and between community relation probability q
                sizes = [100, 100, 100]
                P = [
                    [p, q, q],
                    [q, p, q],
                    [q, q, p]
                ]
                colsize = 20
                def stage(text):
                    print(text + (colsize - len(text))*" ", end="")
                def back():
                    print(colsize*"\b", end="")
                stage("Creating SBM")
                samples = [] 
                for i in range(10):
                    sbm = nx.stochastic_block_model(sizes, P)
                    back()
                    # calculate corruptability of the graph
                    stage("Calculating cor.")
                    corruptability = calculate_graph_corruptability(sbm, 3, use_modified_lfm=True)
                    samples.append(corruptability)
                corruptabilities.append(np.mean(samples))
                back()
                print("\r", end="" )
            # decrease outliers 
            for i in range(1, len(corruptabilities)-1):
                corruptabilities[i] = (corruptabilities[i-1] + corruptabilities[i] + corruptabilities[i+1]) / 3
            print()
            plt.plot(qs, corruptabilities, label=f"p={p}")
        print()
        plt.legend()
        print("Saving plot")
        plt.show()
        # plt.savefig("corruptability_sbm.png")
    def calculate_corruptability_for_popular_datasets(self):
        import pandas as pd 
        from vertex_voyage.partitioning import calculate_graph_corruptability
        import networkx as nx 
        import numpy as np 
        datasets = {
            "Zaharijev karate klub": nx.karate_club_graph(),
            "Les Mis": nx.les_miserables_graph(),
            "Davis southern women": nx.davis_southern_women_graph(),
            "Florentine families": nx.florentine_families_graph()
        }
        data = []
        for name, dataset in datasets.items():
            print(f"Calculating corruptability for {name}")
            corruptability = calculate_graph_corruptability(dataset, 3)
            data.append({
                "Mreža": name,
                "Koruptabilnost": corruptability
            })
        df = pd.DataFrame(data)
        return df

    def calculate_partitioning_time_for_popular_datasets(self):
        import pandas as pd 
        from vertex_voyage.partitioning import partition_graph
        import networkx as nx 
        import numpy as np 
        import time 
        datasets = {
            "Zaharijev karate klub": nx.karate_club_graph(),
            "Les Mis": nx.les_miserables_graph(),
            "Davis southern women": nx.davis_southern_women_graph(),
            "Florentine families": nx.florentine_families_graph()
        }
        data = []
        for name, dataset in datasets.items():
            print(f"Calculating partitioning time for {name}")
            start = time.time()
            partition_graph(dataset, 3)
            end = time.time()
            data.append({
                "Mreža": name,
                "Vreme particionisanja": end-start,
                "Broj čvorova": dataset.number_of_nodes(),
                "Broj grana": dataset.number_of_edges()
            })
        df = pd.DataFrame(data)
        return df.to_html(index=False)
    
    def calculate_partitioning_time_for_sbm(self):
        import pandas as pd 
        from vertex_voyage.partitioning import partition_graph
        import networkx as nx 
        import numpy as np 
        import time 
        data = []
        p = .1
        q = .01
        x = [] 
        y = [] 
        for i in range(1, 11):
            print(f"Calculating partitioning time for SBM {i}")
            sizes = [100*i, 100*i, 100*i]
            x.append(100*i)
            P = [
                [p, q, q],
                [q, p, q],
                [q, q, p]
            ]
            sbm = nx.stochastic_block_model(sizes, P)
            start = time.time()
            partition_graph(sbm, 3)
            end = time.time()
            original_lfm_time = end-start
            y.append(original_lfm_time)
            start = time.time()
            partition_graph(sbm, 3, use_modified_lfm=True)
            end = time.time()
            modified_lfm_time = end-start
            y.append(end-start)
            data.append({
                "Veličina zajednice": 100*i,
                "Vreme particionisanja (originalni LFM)": original_lfm_time,
                "Vreme particionisanja (modifikovani LFM)": modified_lfm_time,
                "Broj čvorova": sbm.number_of_nodes(),
                "Broj grana": sbm.number_of_edges()
            })
        df = pd.DataFrame(data)
        return df.to_html(index=False)
    
    def analyze_corruptability_modified_lfm(self):
        import pandas as pd 
        from vertex_voyage.partitioning import calculate_graph_corruptability
        import networkx as nx 
        import numpy as np 
        import matplotlib.pyplot as plt
        thresholds = list(np.linspace(0.1, 0.9, 10))
        corruptabilities = [] 
        for threshold in thresholds:
            print(f"Calculating corruptability for threshold {threshold}")
            corruptability = calculate_graph_corruptability(nx.karate_club_graph(), 3, use_modified_lfm=True, threshold=threshold)
            corruptabilities.append(corruptability)
        plt.title("Koruptabilnost Karate kluba u zavisnosti od praga modifikovanog LFM algoritma (threshold)")
        plt.xlabel("Threshold")
        plt.ylabel("Koruptabilnost")
        plt.plot(thresholds, corruptabilities)
        plt.show()
    
    def merge_csv(self, first: str, second: str, column: str, renamed_column: str):
        import pandas as pd 
        first = pd.read_csv(first)
        second = pd.read_csv(second)
        first[renamed_column] = second[column]
        return first.to_html(index=False)

    def analyze_time(self, single_results: str, multi_results: str):
        import json 
        import pandas as pd 
        single_results = json.load(open(single_results, "r"))
        multi_results = json.load(open(multi_results, "r"))
        x_label = [k for k in single_results[0].keys() if k != "result" and k != "time"][0]
        y_label = "Vreme"
        x = []
        y1 = []
        y2 = []
        for single_result, multi_result in zip(single_results, multi_results):
            x.append(single_result[x_label])
            y1.append(single_result["result"]["time"])
            y2.append(multi_result["result"]["time"])
        df = pd.DataFrame({
            x_label: x,
            y_label + " (serijska obrada)": y1,
            y_label + " (paralelna obrada)": y2
        })
        return df.to_html(index=False)
    
    def get_time_data(self, results: str):
        import json 
        import pandas as pd 
        results = json.load(open(results, "r"))
        x_label = [k for k in results[0].keys() if k != "result" and k != "time"][0]
        y_label = "Vreme"
        x = []
        y = []
        for result in results:
            x.append(result[x_label])
            y.append(result["result"]["time"])
        df = pd.DataFrame({
            x_label: x,
            y_label: y
        })
        return df

COMMAND_CLASSES = ["Client"]