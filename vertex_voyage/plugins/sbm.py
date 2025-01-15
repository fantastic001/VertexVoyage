import networkx as nx 
import numpy as np 


class LinearCongruentialGenerator:
    def __init__(self, seed, a, c, m):
        """
        Initialize the Linear Congruential Generator.

        Parameters:
        seed (int): The initial seed value (x0).
        a (int): The multiplier.
        c (int): The increment.
        m (int): The modulus.
        """
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m

    def _modular_exponentiation(self, base, exp, mod):
        """Efficiently compute (base^exp) % mod."""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp // 2
            base = (base * base) % mod
        return result

    def get_number(self, i):
        """
        Compute the ith number in the sequence without iteration.

        Parameters:
        i (int): The index of the number to compute (0-based).

        Returns:
        int: The ith number in the sequence.
        """
        if i == 0:
            return self.seed % self.m

        # Compute a^i % m
        a_i_mod_m = self._modular_exponentiation(self.a, i, self.m)

        if self.a == 1:
            # Special case when a == 1
            sum_mod_m = (i * self.c) % self.m
        else:
            # Compute the sum of the geometric series
            numerator = (1 - a_i_mod_m) % self.m
            denominator = (1 - self.a) % self.m
            # Modular multiplicative inverse of denominator
            denominator_inv = pow(denominator, -1, self.m)
            sum_mod_m = (numerator * denominator_inv) % self.m

        # Compute the ith number
        x_i = (a_i_mod_m * self.seed + self.c * sum_mod_m) % self.m
        return x_i



class SBMNetwork:
    def __init__(self, sizes, probabilities, lcg_params):
        """
        Initialize the Stochastic Block Model Network.

        Parameters:
        sizes (list): Sizes of the blocks (communities).
        probabilities (list of lists): Probability matrix for edges between blocks.
        lcg_params (dict): Parameters for the Linear Congruential Generator (seed, a, c, m).
        """
        self.sizes = sizes
        self.probabilities = probabilities
        self.node_to_block = []
        self.total_nodes = sum(sizes)
        self.lcg = LinearCongruentialGenerator(**lcg_params)

        # Assign nodes to blocks
        start = 0
        for block_id, size in enumerate(sizes):
            self.node_to_block.extend([block_id] * size)

    def _get_edge_index(self, u, v):
        """Compute the unique index for edge (u, v) in the random process."""
        if u > v:
            u, v = v, u  # Ensure u < v for undirected edges
        return u * self.total_nodes + v

    def edge_exists(self, u, v):
        """
        Determine if an edge exists between nodes u and v based on the probability matrix.

        Parameters:
        u (int): First node.
        v (int): Second node.

        Returns:
        bool: True if the edge exists, False otherwise.
        """
        if u == v:
            return False  # No self-loops

        block_u = self.node_to_block[u]
        block_v = self.node_to_block[v]
        probability = self.probabilities[block_u][block_v]

        edge_index = self._get_edge_index(u, v)
        random_value = self.lcg.get_number(edge_index) / self.lcg.m

        return random_value < probability

    def get_neighbors(self, node):
        """
        Get the neighbors of a given node.

        Parameters:
        node (int): The node for which to find neighbors.

        Returns:
        list: A list of neighboring nodes.
        """
        neighbors = []
        for other_node in range(self.total_nodes):
            if node != other_node and self.edge_exists(node, other_node):
                neighbors.append(other_node)
        return neighbors


def generate_big_network():
    nodes = [1000 for i in range(10)]
    communities = 10
    B = np.zeros((communities, communities))
    for i in range(communities):
        for j in range(communities):
            if i == j:
                B[i][j] = 0.5
            else:
                B[i][j] = 0.1
    return nx.stochastic_block_model(sizes=nodes, p=B)

if __name__ == "__main__":
    generate_big_network()