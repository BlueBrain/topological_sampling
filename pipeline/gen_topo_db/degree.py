import numpy as np


def make_compute_degree(axis):
    def compute(tribes, adj_matrix, conv, precision):
        return np.array(adj_matrix.sum(axis=axis)).flatten().tolist()

    return compute
