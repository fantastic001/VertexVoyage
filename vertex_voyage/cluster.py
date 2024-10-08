
# import zookeeper client 
from kazoo.client import KazooClient
import os 
import random 
import re
from kazoo.exceptions import NodeExistsError
import threading


zk = None 
USE_ZK = os.getenv('USE_ZK', '1').lower() == '1'
ZK_PATH = '/vertex_voyage'
ZK_NODE_PATH = ZK_PATH + '/nodes/'
ENV_NODE_NAME = os.getenv('NODE_NAME', random.randbytes(4).hex())
def get_zk_client():
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
def register_node():
    def f():
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

def get_nodes():
    if not USE_ZK:
        return ["localhost"]
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return sorted(nodes)

def get_node_data(node):
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    node_path = ZK_NODE_PATH + node
    data, stat = zk.get(node_path)
    return data

def get_ip_by_index(index):
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    node = get_node_by_index(index)
    data = get_node_data(node)
    return data.split()[1].decode()

def get_node_index(node):
    if not USE_ZK:
        return 0
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes.index(node)

def get_node_by_index(index):
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes[index]

def get_leader():
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

def get_current_node():
    if not USE_ZK:
        return os.getenv('NODE_ADDRESS', "")
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    for node in nodes:
        data = get_node_data(node)
        if data.split()[1].decode() == os.getenv('NODE_ADDRESS'):
            return node
    return None

def do_rpc(node_index, method_name, **kwargs):
    print(f"do_rpc({node_index}, {method_name}, {kwargs})")
    ip = get_ip_by_index(node_index)
    from xmlrpc.client import ServerProxy
    s = ServerProxy(f'http://{ip}:8000')
    return s.execute(method_name, kwargs)

def get_node_index_by_ip(ip):
    if not USE_ZK:
        return 0
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    for i, node in enumerate(nodes):
        data = get_node_data(node)
        if data.split()[1].decode() == ip:
            return i
    return None


def do_rpc_to_leader(method_name, **kwargs):
    if not USE_ZK:
        return do_rpc(0, method_name, **kwargs)
    leader = get_leader()
    leader_index = get_node_index(leader)
    return do_rpc(leader_index, method_name, **kwargs)

def is_leader():
    if not USE_ZK:
        return True
    print("Leader: ", get_leader())
    print("Current: ", get_current_node())
    return get_leader() == get_current_node()

def do_rpc_client(ip, method_name, **kwargs):
    from xmlrpc.client import ServerProxy, Fault
    s = ServerProxy(f'http://{ip}:8000')
    try:
        return s.execute(method_name, kwargs)
    except Fault as err:
        return {
            "error": err.faultString,
            "code": err.faultCode
        }