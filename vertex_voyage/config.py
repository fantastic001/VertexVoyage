import os 
import json
import importlib 
import sys 

class VertexVoyageConfigurationError(Exception):
    pass

class VertexVoyageConflictError(Exception):
    pass

PROJECT_NAME = "vertex_voyage"
PROJECT_VERSION = "0.1.0"
PROJECT_ENVVAR_PREFIX = "VERTEX_VOYAGE"
DEFAULT_PLUGINS = [
    "vertex_voyage.plugins.greeting",
    "vertex_voyage.plugins.config_commands",
    "vertex_voyage.plugins.plugin_manager",
]

def get_config_location():
    return os.environ.get(PROJECT_ENVVAR_PREFIX + "_CONFIG", os.path.join(os.path.expanduser("~"), ".vertex_voyage", "config.json"))

def get_config(key: str, default, doc, expected_type=str):
    config = {} 
    envvar = PROJECT_ENVVAR_PREFIX + "_" + key.upper()
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

def get_config_int(key: str, default, doc) -> int:
    return get_config(key, default, doc, expected_type=int)

def get_config_str(key: str, default, doc) -> str:
    return get_config(key, default, doc, expected_type=str)

def get_config_bool(key: str, default, doc) -> bool:
    return get_config(key, default, doc, expected_type=bool)

def get_config_list(key: str, default, doc) -> list:
    return get_config(key, default, doc, expected_type=list)

def get_config_dict(key: str, default, doc) -> dict:
    return get_config(key, default, doc, expected_type=dict)

def get_config_float(key: str, default, doc) -> float:
    return get_config(key, default, doc, expected_type=float)

def set_config(key: str, value):
    config_location = get_config_location()
    config = {}
    if os.path.exists(config_location):
        with open(config_location) as f:
            config = json.load(f)
    config[key] = value
    with open(config_location, "w") as f:
        json.dump(config, f)

def get_search_path():
    return get_config_list("plugin_search_path", [
        "/usr/local/share/vertex_voyage/plugins",
        "/usr/share/vertex_voyage/plugins",
        os.path.join(os.path.expanduser("~"), ".vertex_voyage", "plugins")
    ], "List of plugin search paths")

get_disabled_plugins = lambda: get_config_list("disabled_plugins", [], "List of plugins to disable")

def list_plugins():
    plugins = DEFAULT_PLUGINS
    plugins += get_config_list("plugins", [], "List of plugins to load")
    disabled_plugins = get_disabled_plugins()
    plugins = [plugin for plugin in plugins if plugin not in disabled_plugins]
    return list(set(plugins))

def load_plugins():
    oldpath = sys.path
    sys.path = get_search_path() + sys.path
    plugins = list_plugins()
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
            getattr(plugin, method_name)(*args, **kwargs)
    

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