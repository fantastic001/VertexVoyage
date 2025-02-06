
import inspect 
import os
from vertex_voyage.config import get_classes_inheriting

class BaseModel:
    def __init__(self):
        pass

    def valid(self):
        return False 
    def fit(self):
        pass
    def ready(self):
        return False 
    def run(self, input):
        return None 

    
    def key(self):
        return None

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
        if method not in ["valid", "fit", "ready", "run", "key"]:
            actions[method] = {
                "parameters": get_parameters(model, method),
            }
    return actions

def get_input_type(model: BaseModel):
    parameters =  get_parameters(model, "run")
    if len(parameters) == 0:
        return None
    return list(parameters.values())[0]

def get_output_type(model: BaseModel):
    return get_return_type(model, "run")

def get_model_info(model: BaseModel):
    return {
        "input": get_input_type(model),
        "output": get_output_type(model),
        "actions": get_actions(model),
        "key": model.key(),
        "class": model.__class__.__name__,
        "parameters": get_parameters(model, "fit"),
    }

def apply_actions(model: BaseModel, actions: list):
    for action in actions:
        method = action['method']
        parameters = action['parameters']
        model = getattr(model, method)(**parameters)
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

def get_model_classes():
    return get_classes_inheriting(BaseModel)

def get_model_names():
    return [cls.__name__ for cls in get_model_classes()]