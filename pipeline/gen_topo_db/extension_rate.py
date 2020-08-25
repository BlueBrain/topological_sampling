import numpy as np
import progressbar
from toposample.indexing import GidConverter

def compute(tribes, adj_matrix, precision):

    extensions = []
    conv = GidConverter(tribes)
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        extension_submat = adj_matrix[tribe_ids]

        # Get all the neighbours of all the vertices in the tribe
        all_neighbours = np.nonzero(np.sum(extension_submat,axis=0))[0]

        # Remove tribe itself to get the extension rate
        extension_rate = len(all_neighbours) - len(tribe)

        # Something is funky if this trips
        assert extension_rate > 0

        extensions.append(extension_rate)
        
    return extensions
