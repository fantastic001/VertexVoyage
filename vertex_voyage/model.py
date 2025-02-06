
import inspect 
import os
from vertex_voyage.config import get_classes_inheriting
import yaml 
from vertex_voyage.data import NewColumns, Product, SatisfiesAll, SchemaCheck, Table, RemappedTable
from hashlib import sha256
import json 
class BaseModel:
    def __init__(self):
        pass

    def create(self) -> 'BaseModel':
        return None 

    def valid(self):
        return False 
    def fit(self):
        pass
    def ready(self):
        return False 
    def expects(self) -> SchemaCheck:
        return SchemaCheck()
    def run(self, input: Table) -> Table:
        return None 
    def produces(self) -> Product:
        return Product()

    def get_data(self):
        return None 
    
    def key(self):
        return sha256(json.dumps({
            "class": self.__class__.__name__,
            "data": self.get_data()
        }).encode()).hexdigest()

def is_estimable(model: BaseModel):
    return isinstance(model, BaseModel) and hasattr(model, 'fit') and model.valid() and not model.ready()

def is_runnable(model: BaseModel):
    return is_estimable(model) and hasattr(model, 'run') and model.valid() and model.ready()

def get_parameters(model: BaseModel, method='fit'):
    """
    Returns dictionary of parameters of the model and their types.

    Types are determined by looking at type annotation of the fit method.
    """
    sig = inspect.signature(getattr(model, method))
    return {name: param.annotation for name, param in sig.parameters.items() if param.annotation != inspect.Parameter.empty}

def get_return_type(model: BaseModel, method='run'):
    return getattr(getattr(model, method), '__annotations__', {}).get('return', None)

def get_actions(model: BaseModel):
    actions = {} 
    for method, _ in inspect.getmembers(model, predicate=inspect.ismethod):
        if method.startswith("__"):
            continue
        if method not in ["create", "valid", "fit", "ready", "run", "key", "get_data", "produces", "expects"]:
            actions[method] = {
                "parameters": get_parameters(model, method),
            }
    return actions



def get_model_info(model: BaseModel):
    return {
        "input": model.expects(),
        "output": model.produces(),
        "actions": get_actions(model),
        "key": model.key(),
        "class": model.__class__.__name__,
        "parameters": get_parameters(model, "fit"),
        "data": model.get_data(),
        "constructor": get_parameters(model, "create"),
    }

def apply_actions(model: BaseModel, actions: list):
    for action in actions:
        method = action.pop('method')
        model = getattr(model, method)(**action)
    return model


def load_model(path: str):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model(model: BaseModel, destination_path: str):
    import pickle
    path = os.path.join(destination_path, model.key() + ".pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def construct_model(name: str, actions: list):
    for cls in get_classes_inheriting(BaseModel):
        if cls.__name__ == name:
            return apply_actions(cls(), actions)
    return None

def construct_from_yml(path: str):
    data = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return construct_model(data["model"], data.get("actions", []))

def get_model_classes():
    return get_classes_inheriting(BaseModel)

def get_model_names():
    return [cls.__name__ for cls in get_model_classes()]

class Remapping(BaseModel):
    def __init__(self, mapping = {}):
        self.mapping = mapping
    
    def map(self, column: str, new_column:str) -> 'Remapping':
        return Remapping(mapping={**self.mapping, column: new_column})

    def run(self, input: Table) -> Table:
        return RemappedTable(input, self.mapping)

    def produces(self) -> Product:
        return NewColumns(columns=self.mapping)

    def expects(self) -> SchemaCheck:
        return SatisfiesAll([HasColumn(c) for c in self.mapping.keys()])

    def get_data(self):
        return self.mapping

class ModelSpec(BaseModel):
    def __init__(self, path: str = None):
        self.path = path
        if path is not None:
            self.model = construct_from_yml(path)
        else:
            self.model = None

    def create(self, path: str) -> 'BaseModel':
        return ModelSpec(path)

    def valid(self):
        return self.model is not None and self.model.valid()
    def fit(self, data: Table):
        if self.model is not None:
            self.model.fit(data)
    def ready(self):
        return self.model is not None and self.model.ready()
    def expects(self) -> SchemaCheck:
        return self.model.expects()
    def run(self, input: Table) -> Table:
        return self.model.run(input)
    def produces(self) -> Product:
        return self.model.produces()

    def get_data(self):
        return {
            "path": self.path
        }
    