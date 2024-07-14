
# import zookeeper client 
from kazoo.client import KazooClient
import os 
import random 

zk = None 

ZK_PATH = '/vertex_voyage'
ZK_NODE_PATH = ZK_PATH + '/nodes/'
ENV_NODE_NAME = os.getenv('NODE_NAME', random.randbytes(4).hex())
def get_zk_client():
    global zk 
    if zk is None:
        zk = KazooClient(hosts='localhost:2181')
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

def get_node_index(node):
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes.index(node) + 1

def get_node_by_index(index):
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes[index - 1]

def get_leader():
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    if len(nodes) > 0:
        return nodes[0]
    else:
        return None

def get_current_node():
    zk = get_zk_client()
    nodes = zk.get_children(ZK_NODE_PATH)
    return nodes[-1]