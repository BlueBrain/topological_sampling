import numpy as np
import progressbar
import scipy.linalg


def compute(tribes, adj_matrix, conv, precision):
    import networkx as nx

    spectra = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]
        G = nx.from_scipy_sparse_matrix(adj_submat, create_using=nx.DiGraph)

        # Find the largest connected component of the graph
        largest = max(nx.strongly_connected_components(G), key=len)

        if len(largest) <= 2:  # Needs at least a certain size...
            spectra.append([])
        else:
            # Adjacency matrix of the tribe's strong component
            tribe_strong_adj_submat = nx.to_numpy_array(G.subgraph(largest), dtype='int8')

            # Make a diagonal matrix of inverses of outdegrees in the tribe
            diag_outdegree_inverses = np.diagflat(np.power(np.sum(tribe_strong_adj_submat, axis=1).astype(float), -1))

            # The transition probability matrix
            tr_prob = diag_outdegree_inverses @ tribe_strong_adj_submat          

            # Find the eigenvalues
            eig = scipy.linalg.eig(tr_prob)[0]

            # Order the non-zero eigenvalues and round to desired precision
            spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
            spectra.append(spectrum)
            
    return spectra
