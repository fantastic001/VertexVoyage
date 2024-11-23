
from vertex_voyage.command_executor import * 
from vertex_voyage.cluster import register_node, get_binding_port
from vertex_voyage.config import notify_plugins, get_config_int

notify_plugins("node_starting")
register_node()
notify_plugins("node_started")

command_executor_rpc(get_classes("vertex_voyage"), get_binding_port())