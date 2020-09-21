import numpy as np
import progressbar
import scipy

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
            # Compute the Chung's laplacian matrix of tribe's largest connected component
            L = nx.directed_laplacian_matrix(G.subgraph(largest))

            # Find the eigenvalues
            eig = scipy.linalg.eig(L)[0]

            # Order the non-zero eigenvalues and round to desired precision
            spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
            spectra.append(spectrum)
            
    return spectra
