import numpy as np
import progressbar


def compute(tribes, adj_matrix, conv, precision):
    import networkx as nx

    rel_boundaries = []
    pbar = progressbar.ProgressBar()
    G_full = nx.from_scipy_sparse_matrix(adj_matrix)

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        G = nx.from_scipy_sparse_matrix(adj_submat, create_using=nx.DiGraph)
        edges_in_tribe = G.number_of_edges()
        if edges_in_tribe == 0:
            rel_boundaries.append(0)
        else:
            # Get the edge boundary of a tribe within the full circuit
            boundary = nx.algorithms.boundary.edge_boundary(G_full, tribe_local_indices)

            rel_boundaries.append(np.round(len(list(boundary)) / edges_in_tribe, precision))

    return rel_boundaries
