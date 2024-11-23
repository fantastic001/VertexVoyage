
from vertex_voyage.command_executor import * 
from vertex_voyage.cluster import register_node, get_binding_port
from vertex_voyage.config import notify_plugins, get_config_int
from vertex_voyage import ControlInterface
import threading 
def main():
    notify_plugins("node_starting")
    register_node()
    notify_plugins("node_started")
    notify_plugins("register_commands", ControlInterface())
    command_executor_rpc(get_classes("vertex_voyage") + ControlInterface.additional_classes, get_binding_port())
if __name__ == '__main__':
    main()