import numpy as np
import progressbar
from pyflagsercontain import flagser_count


def compute(tribes, adj_matrix, conv, precision):

    # Transitive clustering coefficients of chiefs
    trccs = []

    simplexcontainment = flagser_count(adj_matrix)
    indegs = np.array(adj_matrix.sum(axis=0))[0]
    outdegs = np.array(adj_matrix.sum(axis=1))[:, 0]
    totdegs = np.array((adj_matrix + adj_matrix.transpose()).sum(axis=0))[0]
    recip_degs = indegs + outdegs - totdegs

    # This assumes tribes are in the order of adjacency matrix indexing
    pbar = progressbar.ProgressBar(maxval=len(indegs))
    for indeg, outdeg, totdeg, recip_deg, smplxcont in pbar(zip(indegs, outdegs,
                                                                totdegs, recip_degs,
                                                                simplexcontainment)):
        denom = totdeg*(totdeg-1)-(indeg*outdeg+recip_deg)

        if denom != 0 and len(smplxcont) > 2:
            # smplxcont[i][2] gives the number of directed 2-cliques that vertex i belongs to
            parameter = np.round(smplxcont[2] / denom, precision)
        else:
            parameter = 0
        trccs.append(parameter)

    return trccs
