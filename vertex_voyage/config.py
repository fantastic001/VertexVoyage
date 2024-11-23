import os 
import json
import importlib 
import sys 

class VertexVoyageConfigurationError(Exception):
    pass

class VertexVoyageConflictError(Exception):
    pass

def get_config_location():
    return os.environ.get("VERTEX_VOYAGE_CONFIG", os.path.join(os.path.expanduser("~"), ".vertex_voyage", "config.json"))

def get_config(key: str, default, expected_type=str):
    config = {} 
    envvar = "VERTEX_VOYAGE_" + key.upper()
    if envvar in os.environ:
        if expected_type == bool:
            return os.environ[envvar].lower() in ["true", "1", "yes", "y", "YES", "Y", "True", "TRUE"]
        if expected_type == list:
            return os.environ[envvar].split(" ")
        return expected_type(os.environ[envvar])
    default_config_location = os.path.join(os.path.expanduser("~"), ".vertex_voyage", "config.json")
    config_location = os.environ.get("VERTEX_VOYAGE_CONFIG", default_config_location)
    if not os.path.exists(config_location) and config_location != default_config_location:
        raise VertexVoyageConfigurationError(f"Configuration file {config_location} does not exist")
    if not os.path.exists(config_location):
        return expected_type(default)
    with open(config_location) as f:
        config = json.load(f)
    return expected_type(config.get(key, default))

def get_config_int(key: str, default) -> int:
    return get_config(key, default, expected_type=int)

def get_config_float(key: str, default) -> float:
    return get_config(key, default, expected_type=float)

def get_config_bool(key: str, default) -> bool:
    return get_config(key, default, expected_type=bool)

def get_config_str(key: str, default) -> str:
    return get_config(key, default, expected_type=str)

def get_config_list(key: str, default) -> list:
    return get_config(key, default, expected_type=list)

def set_config(key: str, value):
    config_location = get_config_location()
    config = {}
    if os.path.exists(config_location):
        with open(config_location) as f:
            config = json.load(f)
    config[key] = value
    with open(config_location, "w") as f:
        json.dump(config, f)

def load_plugins():
    search_path = get_config_list("plugin_search_path", [])
    oldpath = sys.path
    sys.path = search_path + sys.path
    plugins = get_config_list("plugins", [])
    result = []
    for plugin in plugins:
        try:
            result.append((plugin, importlib.import_module(plugin)))
        except ImportError as e:
            print(f"Error loading plugin {plugin}: {e}")
    sys.path = oldpath
    return result

def notify_plugins(method_name, *args, **kwargs):
    plugins = load_plugins()
    for name, plugin in plugins:
        if hasattr(plugin, method_name):
            return getattr(plugin, method_name)(*args, **kwargs)
    

def get_plugin_result(method_name, *args, **kwargs):
    plugins = load_plugins()
    found_results = []
    for name, plugin in plugins:
        if hasattr(plugin, method_name):
            found_results.append((name, getattr(plugin, method_name)(*args, **kwargs)))
    if len(found_results) == 0:
        return None
    if len(found_results) == 1:
        return found_results[0][1]
    raise VertexVoyageConflictError(f"Multiple plugins returned results for {method_name}: {found_results}")

def pluggable(func):
    func_name = func.__name__
    def wrapper(*args, **kwargs):
        result = get_plugin_result(func_name, *args, **kwargs)
        if result is not None:
            return result
        return func(*args, **kwargs)
    return wrapper