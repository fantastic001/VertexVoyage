
from vertex_voyage.command_executor import command_executor_main
from vertex_voyage.config import get_classes_inheriting
import yaml
from vertex_voyage.data_source import DataSource, load_data_source_from_yaml, get_data_source_info
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

class Simulator():
    def models(self):
        return get_model_names()
    def info(self, model_name: str):
        models = get_model_classes()
        for cls in models:
            if cls.__name__ == model_name:
                return get_model_info(cls())
        return None

    def create(self, model_name: str, path: str):
        models = get_model_classes()
        for cls in models:
            if cls.__name__ == model_name:
                return yaml.dump(
                    {
                        "model": model_name,
                        "actions": []
                    },
                    open(path, "w"),
                )
        return None
    
    def load(self, path: str):
        return get_model_info(construct_from_yml(path))

    def source(self, path: str):
        return get_data_source_info(load_data_source_from_yaml(path))

if __name__ == "__main__":
    command_executor_main([Simulator])