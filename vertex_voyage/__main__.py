
from vertex_voyage.command_executor import * 
from vertex_voyage.cluster import register_node, get_binding_port
from vertex_voyage.config import notify_plugins, get_config_int

notify_plugins("node_starting")
register_node()
notify_plugins("node_started")


additional_classes = [] 
class ControlInterface:
    def add_command_class(self, cls):
        additional_classes.append(cls)

notify_plugins("register_commands", ControlInterface())

command_executor_rpc(get_classes("vertex_voyage") + additional_classes, get_binding_port())