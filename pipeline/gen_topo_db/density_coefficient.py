import numpy as np
import progressbar
from pyflagsercontain import flagser_count


def compute(tribes, adj_matrix, precision):

    # Density coefficients
    dcs = []
    graph_size = len(adj_matrix)
    simplexcontainment = flagser_count(adj_matrix)

    # This assumes tribes are in the order of adjacency matrix indexing
    for i in pbar(len(tribes)):
        counts = simplexcontainment[i]

        dc_list = []

        for k in range(2,len(counts)):
            parameter_at_k = k*counts[k]/((k+1)*(graph_size-k)*counts[k-1]) if counts[k-1] != else 0
            dc_list.append(parameter_at_k)

        dcs.append(dc_list)

    return dcs