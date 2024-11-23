
from vertex_voyage import cluster 
from vertex_voyage import ControlInterface
import time
import os 
import sys 
import threading
def register_commands(control_interface: ControlInterface):
    def f():
        time.sleep(5)
        print("Running smoke test")
        cluster.do_rpc_client("localhost", "create_graph", name="test")
        graphs = cluster.do_rpc_client("localhost", "list")
        if "test" in graphs:
            print("Smoke test passed")
            print("Killing the process gracefully")
            os.kill(os.getpid(), 15)
    control_interface.start_background_thread(f)
