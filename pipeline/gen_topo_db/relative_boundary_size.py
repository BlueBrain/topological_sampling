import numpy as np
import progressbar
from toposample.indexing import GidConverter

def compute(tribes, adj_matrix, precision):
    import networkx as nx

    rel_boundaries = []
    pbar = progressbar.ProgressBar()
    conv = GidConverter(tribes.index)

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = np.array(adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)].todense())
        G = nx.from_numpy_array(adj_submat, create_using=nx.DiGraph)
        edges_in_tribe = G.number_of_edges()

        # Get the edge boundary of a tribe within the full circuit
        boundary = nx.algorithms.boundary.edge_boundary(G,tribe_local_indices)

        rel_boundaries.append(np.round(len(list(boundary))/edges_in_tribe,precision))

    return rel_boundaries
