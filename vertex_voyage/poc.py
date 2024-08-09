
import networkx as nx 
import cdlib 
import cdlib.algorithms
import cdlib.utils
import numpy as np
import vertex_voyage.node2vec as nv 
import sklearn
from vertex_voyage.cluster import *




def calculate_graph_similarity(g1, g2):
    """
    Calculate the similarity between two graphs
    """
    # get the number of nodes in each graph
    n1 = g1.number_of_nodes()
    n2 = g2.number_of_nodes()

    # get the number of edges in each graph
    e1 = g1.edges()
    e2 = g2.edges()

    # get the number of common edges between the two graphs
    common_edges = len(set(g1.edges()).intersection(set(g2.edges())))

    # calculate the similarity between the two graphs
    similarity = common_edges / len(set(e1).union(set(e2)))

    return similarity

def partition_graph(G):
    """
    Uses LFM to partition graph into overlapping partitions
    """
    # create a LFM object
    communities = cdlib.algorithms.lfm(G, 1)

    return communities.communities

def get_embedding(G):
    """
    Get the embedding of a graph
    """
    # create a Node2Vec object
    node2vec = nv.Node2Vec(dim=4, walk_size=10, n_walks=100, window_size=5)

    # fit the Node2Vec object to the graph
    node2vec.fit(G)

    return node2vec
    

def kmeans(nodes):
    """
    Perform k-means clustering on a set of nodes
    """
    # create a k-means object
    kmeans = sklearn.cluster.KMeans(n_clusters=2)

    # fit the k-means object to the nodes
    km = kmeans.fit(nodes)

    return km.labels_
    

def cluster_similarity(km1, km2):
    """
    Calculate the similarity between two clusterings
    """
    return np.mean(km1 == km2)


def run_poc():
    # get Zachary's Karate Club graph
    G = nx.karate_club_graph()
    print(calculate_graph_similarity(G, G))

    print("Partitions of G:")
    for community in partition_graph(G):
        print(community)

    print("Embedding of G:")
    model = get_embedding(G)
    print("Weights: " + str(model))

    v = []
    for node in G.nodes():
        v.append(model.embed_node(node))
    v = np.array(v)
    print("Embedding of G with vertices : " + str(v))
        

    print("K-means clustering of G:")
    print(kmeans(v))

    print("Cluster similarity:")
    print(cluster_similarity(kmeans(v), kmeans(v)))
