
from vertex_voyage.command_executor import command_executor_main
from vertex_voyage.config import get_classes_inheriting
import yaml
from vertex_voyage.data_source import DataSource, load_data_source_from_yaml, get_data_source_info
from vertex_voyage.simulation import Simulation
from vertex_voyage.model import (
    BaseModel,
    is_estimable,
    is_runnable,
    get_parameters,
    get_return_type,
    get_actions,
    get_model_info,
    get_model_names,
    get_model_classes,
    construct_model,
    construct_from_yml,
)
import yaml 
import os 

def get_full_model_path(model_name: str) -> str:
    return os.path.join("models", model_name.replace(".yml", "") + ".yml")

def get_full_source_path(source_name: str) -> str:
    return os.path.join("data_sources", source_name.replace(".yml", "") + ".yml")

def get_full_input_path(input_name: str) -> str:
    return os.path.join("inputs", input_name.replace(".yml", "") + ".yml")

def is_simulation_root(path: str) -> bool:
    return os.path.exists(os.path.join(path, "main.yml"))

def get_simulation_root(path: str) -> str:
    if is_simulation_root(path):
        return os.path.abspath(path)
    else:
        return get_simulation_root(os.path.dirname(path))

def open_simulation():
    root = get_simulation_root(os.getcwd())
    with open(os.path.join(root, "main.yml"), "r") as f:
        return yaml.safe_load(f)


class Simulator():

    def init(self):
        with open("main.yml", "w") as f:
            yaml.dump({"steps": []}, f)
        for directory in ["models", "data_sources", "inputs"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        return "Initialized simulation"

    def steps(self):
        return [{"model": step["model"], "fit": step["fit"]} for step in open_simulation()["steps"]]
    def models(self):
        return [get_model_info(construct_from_yml(get_full_model_path(model))) for model in os.listdir("models") if model.endswith(".yml")]

    def add_model(self, name: str):
        with open(get_full_model_path(name), "w") as f:
            model = construct_model(name, [])
            possible_actions = get_actions(model)

            yaml.dump({"model": name, "actions": [
                {
                    "method": action,
                    **{param: None for param in get_parameters(model, action)}
                } for action in possible_actions
            ]}, f)
            if "EDITOR" in os.environ:
                os.system(f"$EDITOR {get_full_model_path(name)}")
        return "Added model"
    
    def add_step(self, model: str):
        with open("main.yml", "r") as f:
            data = yaml.safe_load(f)
        data["steps"].append(model)
        with open("main.yml", "w") as f:
            yaml.dump(data, f)
        return "Added step"
    def add_source(self, name: str):
        if name.endswith("csv"):
            with open(get_full_source_path(name.replace(".csv", "")), "w") as f:
                yaml.dump({
                    "class" : "CSVDataSource",
                    "params": {
                        "filename": os.path.abspath(name)
                    }
                }, f)
            return "Added source"
        else:
            raise ValueError("Source type not supported")
    def sources(self):
        return self.simulation().data_sources
    
    def add_input(self, name: str):
        if name.endswith("csv"):
            with open(get_full_input_path(name.replace(".csv", "")), "w") as f:
                yaml.dump({
                    "class" : "CSVDataSource",
                    "params": {
                        "filename": os.path.abspath(name)
                    }
                }, f)
            return "Added input"
        else:
            raise ValueError("Input type not supported")

    def inputs(self):
        return [get_data_source_info(load_data_source_from_yaml(get_full_input_path(input))) for input in os.listdir("inputs") if input.endswith(".yml")]

    def status(self):
        return {
            "models": self.models(),
            "sources": self.sources(),
            "steps": self.steps(),
            "inputs": self.inputs()
        }
    
    def simulation(self) -> Simulation:
        data = open_simulation()
        input_source = data["input"]
        return Simulation(
            data_sources={name.replace(".yml", ""): load_data_source_from_yaml(get_full_source_path(name)) for name in os.listdir("data_sources") if name.endswith(".yml")},
            input=load_data_source_from_yaml(get_full_input_path(input_source)),
            steps=[construct_from_yml(get_full_model_path(step["model"])) for step in data["steps"]],
            params=[step["fit"] for step in data["steps"]],
            name=os.path.basename(get_simulation_root(os.getcwd()))
        )
    
    def full_run(self):
        sim = self.simulation()
        return sim.run()

if __name__ == "__main__":
    command_executor_main([Simulator])