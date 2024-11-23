
# import zookeeper client 
from kazoo.client import KazooClient
import os 
import random 
import re
from kazoo.exceptions import NodeExistsError
import threading
import vertex_voyage.config as cfg 


zk = None 
USE_ZK = os.getenv('USE_ZK', '1').lower() == '1'
ZK_PATH = '/vertex_voyage'
ZK_NODE_PATH = ZK_PATH + '/nodes/'
ENV_NODE_NAME = os.getenv('NODE_NAME', random.randbytes(4).hex())

MPI = None 
USE_MPI = os.getenv('USE_MPI', '0').lower() == '1'
if USE_MPI:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ENV_NODE_NAME = f"node_{rank}"
    print(f"MPI rank: {rank}, size: {size}")
    print("MPI is not yet supported but will be in the future!")
    exit(1)

USE_SLURM = "SLURM_JOB_ID" in os.environ

def get_zk_client():
    if USE_MPI:
        return
    if USE_SLURM:
        return
    if not USE_ZK:
        return
    global zk 
    if zk is None:
        hosts = os.getenv('ZK_HOSTS', 'localhost:2181')
        zk = KazooClient(hosts=hosts)
        zk.start()
    else:
        if not zk.connected:
            zk.start()
    return zk
def zk_callback(event):
    print(f"Zookeeper event: {event}", flush=True)
    if event.type == 'DELETED':
        print("Node deleted")
        register_node()

@cfg.pluggable
def register_node():
    def f():
        if USE_MPI:
            return
        if USE_SLURM:
            return
        if not USE_ZK:
            return
        print("Registering node", flush=True)
        zk = get_zk_client()
        for i in range(5):
            if zk.connected:
                break
            zk.start()
        if not zk.connected:
            raise RuntimeError("Zookeeper client is not connected")
        print("Connected to zookeeper", flush=True)
        if not zk.exists(ZK_PATH):
            try:
                zk.create(ZK_PATH, b'')
            except NodeExistsError as e:
                print(f"Path {ZK_PATH} already exists", flush=True)
        if not zk.exists(ZK_NODE_PATH):
            try:
                zk.create(ZK_NODE_PATH, b'')
            except NodeExistsError as e:
                print(f"Path {ZK_NODE_PATH} already exists", flush=True)
        zk.add_listener(zk_callback)
        node_name = 'node_' + ENV_NODE_NAME
        node_data = ENV_NODE_NAME.encode() 
        # put ip address of current node into node data 
        node_data = node_data + b' ' + os.getenv('NODE_ADDRESS').encode()
        mynodepath = ZK_NODE_PATH + node_name
        if zk.exists(mynodepath):
            zk.set(mynodepath, node_data)
        else:
            zk.create(mynodepath, node_data, ephemeral=True)
            print(f"Registered node {node_name}")
            print(f"Node data: {node_data}")
            print("Current leader: ", get_leader())
            print("Node count: ", len(get_nodes()))
            print("Nodes in cluster: ", get_nodes(), flush=True)
            print("Is leader: ", is_leader(), flush=True)
    # run f in separate thread and wait 10 seconds until it finishes, if it is not completed
    # raise RuntimeError
    t = threading.Thread(target=f)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        raise RuntimeError("Registering node is taking too long")

@cfg.pluggable
def get_nodes():
    if USE_SLURM:
        return os.getenv('SLURM_JOB_NODELIST').split(',')
    if USE_MPI:
        return [f"node_{i}" for i in range(size)]
    if not USE_ZK:
        return ["localhost"]
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return sorted(nodes)

@cfg.pluggable
def get_node_data(node):
    if USE_SLURM:
        return node
    if USE_MPI:
        return f"node_{node} localhost"
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    node_path = ZK_NODE_PATH + node
    data, stat = zk.get(node_path)
    return data

@cfg.pluggable
def get_ip_by_index(index):
    if USE_SLURM:
        return get_nodes()[index]
    if USE_MPI:
        return "localhost"
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    node = get_node_by_index(index)
    data = get_node_data(node)
    return data.split()[1].decode()

@cfg.pluggable
def get_node_index(node):
    if USE_SLURM:
        return get_nodes().index(node)
    if USE_MPI:
        return int(node.split('_')[1])
    if not USE_ZK:
        return 0
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes.index(node)

@cfg.pluggable
def get_node_by_index(index):
    if USE_SLURM:
        return get_nodes()[index]
    if USE_MPI:
        return f"node_{index}"
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes[index]

@cfg.pluggable
def get_leader():
    if USE_SLURM:
        return get_nodes()[0]
    if USE_MPI:
        return "node_0"
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    register_node()
    nodes = sorted(zk.get_children(ZK_NODE_PATH))
    print(f"nodes: {nodes}")
    if len(nodes) > 0:
        return nodes[0]
    else:
        return None

@cfg.pluggable
def get_current_node():
    if USE_SLURM:
        return os.getenv('SLURMD_NODENAME')
    if USE_MPI:
        return f"node_{rank}"
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    for node in nodes:
        data = get_node_data(node)
        if data.split()[1].decode() == os.getenv('NODE_ADDRESS'):
            return node
    return None

def get_binding_port():
    return cfg.get_config_int("port", 8000, "Port to bind to")

@cfg.pluggable
def do_rpc(node_index, method_name, **kwargs):
    if USE_MPI:
        print(f"do_rpc({node_index}, {method_name}, {kwargs})")
        data = {
            "method": method_name,
            "args": kwargs
        }
        if node_index == rank:
            return data
        else:
            comm.send(data, dest=node_index)
            return comm.recv(source=node_index)
    print(f"do_rpc({node_index}, {method_name}, {kwargs})")
    ip = get_ip_by_index(node_index)
    from xmlrpc.client import ServerProxy
    s = ServerProxy(f'http://{ip}:%d' % get_binding_port())
    return s.execute(method_name, kwargs)

@cfg.pluggable
def get_node_index_by_ip(ip):
    if USE_SLURM:
        return get_nodes().index(ip)
    if USE_MPI:
        return 0
    if not USE_ZK:
        return 0
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    for i, node in enumerate(nodes):
        data = get_node_data(node)
        if data.split()[1].decode() == ip:
            return i
    return None


@cfg.pluggable
def do_rpc_to_leader(method_name, **kwargs):
    if not USE_ZK:
        return do_rpc(0, method_name, **kwargs)
    leader = get_leader()
    leader_index = get_node_index(leader)
    return do_rpc(leader_index, method_name, **kwargs)

@cfg.pluggable
def is_leader():
    if not USE_ZK:
        return True
    print("Leader: ", get_leader())
    print("Current: ", get_current_node())
    return get_leader() == get_current_node()

@cfg.pluggable
def do_rpc_client(ip, method_name, **kwargs):
    from xmlrpc.client import ServerProxy, Fault
    s = ServerProxy(f'http://{ip}:{get_binding_port()}')
    try:
        return s.execute(method_name, kwargs)
    except Fault as err:
        return {
            "error": err.faultString,
            "code": err.faultCode
        }