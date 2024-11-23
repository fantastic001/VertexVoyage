import os 
import json
import importlib 
import sys 

class VertexVoyageConfigurationError(Exception):
    pass

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

def load_plugins():
    search_path = get_config_list("plugin_search_path", [])
    oldpath = sys.path
    sys.path = search_path + sys.path
    plugins = get_config_list("plugins", [])
    result = []
    for plugin in plugins:
        try:
            result.append(importlib.import_module(plugin))
        except ImportError as e:
            print(f"Error loading plugin {plugin}: {e}")
    sys.path = oldpath
    return result

def notify_plugins(method_name, *args, **kwargs):
    plugins = load_plugins()
    for plugin in plugins:
        if hasattr(plugin, method_name):
            return getattr(plugin, method_name)(*args, **kwargs)
    

