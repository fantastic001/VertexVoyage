
import os 
import json

class Database:
    def __init__(self, path: str = None) -> None:
        if path is None:
            self.path = os.environ.get("VERTEXVOYAGE_DATABASE_PATH", os.path.join(os.getcwd(), "db"))
        else:
            self.path = path
        self.graph_list_file = os.path.join(self.path, "graph_list.json")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            with open(self.graph_list_file, "w") as f:
                f.write("[]")
    
    def get_graph_list(self) -> list:
        with open(self.graph_list_file, "r") as f:
            return json.load(f)
    
    def add_empty_graph(self, graph_name: str) -> None:
        graph_list = self.get_graph_list()
        graph_list.append({
            "name": graph_name,
            "vertices": [],
            "edges": [],
            "type": "raw"
        })
        with open(self.graph_list_file, "w") as f:
            json.dump(graph_list, f)
    
    def add_nas_graph(self, graph_name: str, nas_url: str) -> None:
        graph_list = self.get_graph_list()
        graph_list.append({
            "name": graph_name,
            "nas": nas_url,
            "type": "nas"
        })
        with open(self.graph_list_file, "w") as f:
            json.dump(graph_list, f)
    