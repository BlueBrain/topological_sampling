import numpy as np
import progressbar
from toposample.indexing import GidConverter
from pyflagsercontain import flagser_count

def compute(tribes, adj_matrix, precision):

    # Transitive clustering coefficients
    trccs = []
    pbar = progressbar.ProgressBar()
    conv = GidConverter(tribes)
    G_full = nx.from_scipy_sparse_matrix(adj_matrix)

    # Compute simplex containments over all vertices
    simplexcontainment = flagser_count(G_full)

    # This assumes tribes are in the order of adjacency matrix indexing
    for i in pbar(range(len(G_full))):

        outdeg = np.count_nonzero(G_full[i])
        indeg = np.count_nonzero(np.transpose(G_full)[i])
        recip_deg = np.count_nonzero(np.logical_and(G_full[i],np.transpose(G_full)[i]))
        totdeg = outdeg+indeg-recip_deg
        denom = totdeg*(totdeg-1)-(indeg*outdeg+recip_deg)

        if denom != 0:
            # simplexcontainment[i][2] gives the number of directed 2-cliques that vertex i belongs to
            parameter = np.round(simplexcontainment[i][2]/denom, precision)
        else:
            parameter = 0
        trccs.append(parameter)

    return trccs