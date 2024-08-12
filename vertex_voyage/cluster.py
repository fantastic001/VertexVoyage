
# import zookeeper client 
from kazoo.client import KazooClient
import os 
import random 
import re

zk = None 

ZK_PATH = '/vertex_voyage'
ZK_NODE_PATH = ZK_PATH + '/nodes/'
ENV_NODE_NAME = os.getenv('NODE_NAME', random.randbytes(4).hex())
def get_zk_client():
    global zk 
    if zk is None:
        hosts = os.getenv('ZK_HOSTS', 'localhost:2181')
        zk = KazooClient(hosts=hosts)
        zk.start()
    else:
        if not zk.connected:
            zk.start()
    return zk

def register_node():
    zk = get_zk_client()
    if not zk.exists(ZK_PATH):
        zk.create(ZK_PATH, b'')
    if not zk.exists(ZK_NODE_PATH):
        zk.create(ZK_NODE_PATH, b'')
    nodes = zk.get_children(ZK_NODE_PATH)
    node_name = 'node_' + str(len(nodes) + 1)
    node_data = ENV_NODE_NAME.encode() 
    # put ip address of current node into node data 
    node_data = node_data + b' ' + os.getenv('NODE_ADDRESS').encode()
    mynodepath = ZK_NODE_PATH + node_name
    if zk.exists(mynodepath):
        zk.set(mynodepath, node_data)
    else:
        zk.create(mynodepath, node_data, ephemeral=True)

def get_nodes():
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes

def get_node_data(node):
    zk = get_zk_client()
    node_path = ZK_NODE_PATH + node
    data, stat = zk.get(node_path)
    return data

def get_ip_by_index(index):
    node = get_node_by_index(index)
    data = get_node_data(node)
    return data.split()[1].decode()

def get_node_index(node):
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes.index(node)

def get_node_by_index(index):
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes[index - 1]

def get_leader():
    zk = get_zk_client()
    register_node()
    nodes = zk.get_children(ZK_NODE_PATH)
    if len(nodes) > 0:
        return nodes[0]
    else:
        return None

def get_current_node():
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
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    for i, node in enumerate(nodes):
        data = get_node_data(node)
        if data.split()[1].decode() == ip:
            return i
    return None


def do_rpc_to_leader(method_name, **kwargs):
    leader = get_leader()
    leader_index = get_node_index(leader)
    return do_rpc(leader_index, method_name, **kwargs)

def is_leader():
    return get_leader() == get_current_node()

def do_rpc_client(ip, method_name, **kwargs):
    from xmlrpc.client import ServerProxy
    s = ServerProxy(f'http://{ip}:8000')
    return s.execute(method_name, kwargs)