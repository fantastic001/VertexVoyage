
from vertex_voyage.command_executor import * 
from vertex_voyage.cluster import register_node

register_node()

command_executor_rpc(get_classes("vertex_voyage"))