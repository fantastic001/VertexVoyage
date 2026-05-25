
from vertex_voyage.temporal import CSVEventSequence, Event, FileEventSequence, FirstN, ListEventSequence
from vertex_voyage.benchmark_base import Benchmark
import pandas as pd 
import os 
import matplotlib.pyplot as plt 


import numpy as np 
import scipy.sparse as sp
import networkx as nx

class NPZLoader:
    """
    Loads npz file and returns edges as event sequence
    """

    def __init__(self, 
        path: str,
        adj_data_label: str = "adj_data",
        adj_indices_label: str = "adj_indices",
        adj_indptr_label: str = "adj_indptr",
        adj_shape_label: str = "adj_shape",
    ):
        self.path = path
        self.adj_data_label = adj_data_label
        self.adj_indices_label = adj_indices_label
        self.adj_indptr_label = adj_indptr_label
        self.adj_shape_label = adj_shape_label

    def __call__(self):
        npz = np.load(self.path)
        matrix = sp.csr_matrix(
            (npz[self.adj_data_label], npz[self.adj_indices_label], npz[self.adj_indptr_label]), 
            shape=npz[self.adj_shape_label]
        )
        g = nx.from_scipy_sparse_array(matrix, create_using = nx.Graph)
        edges = list(g.edges())
        return ListEventSequence([
            Event(src=u, dest=v, timestamp=i)
            for i, (u, v) in enumerate(edges)
        ])




datasets = {
    "Twitch": lambda: FileEventSequence("data/twitch.txt"),
    "Wiki Talks": lambda: FileEventSequence("data/wiki-talks.txt"),
    "AlphaCore": lambda: FileEventSequence("data/AlphaCore.txt"),
    "Reddit": lambda: FileEventSequence("data/reddit.txt"),
    "UK2002": lambda: FileEventSequence("data/uk2002.txt"),
    "Live Journal": lambda: FileEventSequence("data/LiveJournal.txt"),
    "SBM 10M": lambda: FileEventSequence("data/sbm_10M.txt"),
    "Cit-HepPh": lambda: FileEventSequence("data/cit_HepPh.txt"),
    "Cit-HepTh": lambda: FileEventSequence("data/cit_HepTh.txt"),
    "test": lambda: FileEventSequence("data/test.txt"),
    "CITESEER": lambda: FileEventSequence("data/citeseer.txt"),
    "les_mis": lambda: FileEventSequence("data/les_mis.txt"),
    "zachary": lambda: FileEventSequence("data/zachary.txt"),
    # "AstroPh": lambda: FileEventSequence("data/AstroPh.txt"),
    "AstroPh": NPZLoader("data/AstroPh.npz"),
    "AS-Oregon": lambda: FileEventSequence("data/oregon2_as.txt"),
    "DBLP": NPZLoader("data/dblp.npz"),
    "Enron": lambda: FileEventSequence("data/enron.txt"),
    "StackOverflow": lambda: FileEventSequence("data/stackoverflow.txt"),
}


dataset_params = {
    "CITESEER": dict(
        p=.5,
        q=.25,
        dim=50,
    ),
    "AstroPh": dict(
        p=2,
        q=.25,
        dim=50,
    ),
    "Cit-HepPh": dict(
        p=4,
        q=.25,
        dim=100,
    ),
    "Cit-HepTh": dict(
        p=2,
        q=.25,
        dim=100,
    ),
    "zachary": dict(
        p=.25,
        q=4,
        dim=100,
    ),
    "Wiki Talks": dict(
        p=2,
        q=.25,
        dim=100,
    ),
    "Reddit": dict(
        p=2,
        q=.25,
        dim=100,
    ),
    "AlphaCore": dict(
        p=2,
        q=.25,
        dim=100,
    ),
    "AS-Oregon": dict(
        p=0.5,
        q=2,
        dim=128,
    ),
    "DBLP": dict(
        p=0.5,
        q=1,
        dim=25,
    ),
    "Enron": dict(
        p=.5,
        q=1,
        dim=128,
    ),
    "StackOverflow": dict(
        p=1,
        q=2,
        dim=128,
    ),
}
