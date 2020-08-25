import numpy as np


def make_compute_degree(axis):
    def compute(tribes, adj_matrix, precision):
        return np.array(adj_matrix.sum(axis=axis)).flatten().tolist()

    return compute
