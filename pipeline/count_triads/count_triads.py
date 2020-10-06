"""
Topological sampling
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy
import pandas

from scipy import sparse
from scipy.special import comb

from toposample import config
from toposample import TopoData
from toposample.data.data_structures import ConditionCollection, ResultsWithConditions
from toposample.indexing import GidConverter


"""
count_triads.py
Part of the topological sampling pipeline that counts for all generated samples (see sample_tribes-*.py, tribes.json)
the overexpression of triad motifs. It counts the number of different motifs in a sample and also calculates the 
expected number according to two control models and the mean connection probability in the sample.
"""

# Keys: the indices of edges (non-zero entries in a 3x3 connection matrix) after canonical sorting (see below)
# Values: Index of the triad motif (ordered as in Gal et al., 2017)
triad_dict = {
    (1, 6): 0,
    (3, 6): 1,
    (6, 7): 2,
    (3, 6, 7): 3,
    (1, 2, 3): 4,
    (1, 3, 6): 5,
    (1, 5, 6): 6,
    (2, 3, 7): 6,
    (1, 2, 3, 6): 7,
    (1, 3, 5, 6): 8,
    (3, 5, 6, 7): 9,
    (1, 3, 6, 7): 10,
    (1, 2, 3, 6, 7): 11,
    (1, 2, 3, 5, 6, 7): 12
}

# For each triad (ordered as in Gal et al., 2017) how many permutations of it exist
triad_combinations = numpy.array([6, 3, 3, 6, 6, 6, 2, 3, 6, 3, 3, 6, 1])


def read_input(input_config):
    """
    read_input: reads the input required by this stage of the pipeline
    :param input_config: dict; the "inputs" block of the "triads" stage of the common_config
    :return: tribes, TopoData; the various neuron samples
             M, scipy.sparse.csc; the adjacency matrix of the entire circuit
             info, pandas.DataFrame; basic information on all neurons in the circuit
    """
    M = sparse.load_npz(input_config["adjacency_matrix"])
    info = pandas.read_pickle(input_config["neuron_info"])
    tribes = TopoData(input_config["tribes"])
    return tribes, M, info


def submatrix(gids, M, converter):
    """
    submatrix: Gets the submatrix of a neuron sample from the whole adjacency matrix
    :param gids: list; specifies the gids of the sample
    :param M: scipy.sparse.csc; the adjacency matrix of the entire circuit
    :param converter: toposample.indexing.GidConverter
    :return: numpy.array; the adjacency matrix of the neuron sample
    """
    idxx = converter.indices(gids)
    return numpy.array(M[numpy.ix_(idxx, idxx)].todense())


def canonical_sort(M):
    """
    canonical_sort: Sorts the rows/columns of an adjacency matrix canonically, i.e. nodes with highest in-degree first,
    for equal in-degree nodes with highest out-degree first.
    :param M: numpy.array; an adjacency matrix
    :return: a view of M that is canonically sorted
    """
    in_degree = numpy.sum(M, axis=0)
    out_degree = numpy.sum(M, axis=1)
    idx = numpy.argsort(-10 * in_degree - out_degree)
    return M[:, idx][idx]


def identify_motif(M):
    """
    identify_motif: Identifies a fully connected triadic motif (sum of in- and out-degree of each node >= 1)
    :param M: numpy.array; a 3x3 adjacency matrix describing a fully connected motif
    :return: The index of the motif ordered as in Gal et al., 2017
    """
    triad_code = tuple(numpy.nonzero(canonical_sort(M).flatten())[0])
    return triad_dict[triad_code]


def expected_triad_probabilities_er(p):
    """
    expected_triad_probabilities_er: The probabilities of each triadic motif in an Erdos-Renyi control model with
    connection probability p
    :param p: The er connection probability
    :return: The probability of the various triadic motifs, ordered as in Gal et al., 2017. Note: only connectected
    motifs are considered, i.e. motifs with less than 2 connections or only a single bidirectional connection are not
    considered. Therefore, the probabilities do not add up to 1.
    """
    probabilities = numpy.zeros(numpy.max(list(triad_dict.values())) + 1)
    for triad_code, triad_index in triad_dict.items():
        n_con = len(triad_code)
        n_no_con = 6 - n_con  # 6: Highest number of possible connections in a triad
        p_base = (p ** n_con) * ((1 - p) ** n_no_con)
        probabilities[triad_index] = p_base * triad_combinations[triad_index]
        """Note: probabilities do not add up to 1 here,
        because only triads that are fully connected in either direction are considered"""
    return probabilities


def _count_possible_triads_constrained(triad_constr_mat, p):
    """
    :param triad_constr_mat: A 3 x 3 numpy.array of type bool. Entries that are True are considered to have a
    guaranteed connection entries that are 0 have a connection with probability p. Evaluates the probabilities
    of possible triad motifs under these constraints.
    :param p: float
    :return: numpy.array of length 13 with the probabilities for each triad motif under the given constraints.
    """
    wobbles = numpy.nonzero((triad_constr_mat +
                             numpy.eye(triad_constr_mat.shape[0])) == 0)
    wobble_fmt = "{:0" + str(len(wobbles[0])) + "b}"
    probabilities = numpy.zeros(numpy.max(list(triad_dict.values())) + 1)
    for i in range(2 ** len(wobbles[0])):
        wobble_vals = numpy.array(list(map(int, wobble_fmt.format(i))))
        wobble_p = ((1 - p) ** (1 - wobble_vals)) * (p ** wobble_vals)
        wobble_p = numpy.prod(wobble_p)
        triad_constr_mat[wobbles] = wobble_vals
        triad_idx = identify_motif(triad_constr_mat)
        probabilities[triad_idx] += wobble_p
    return probabilities


def expected_triad_probabilities_w_chief(p_rand):
    """What is the expected control probability for each triad, given
    that one of the nodes of each triad is the chief?
    :param p_rand: The probability that a non-chief related edge exists
    """
    probabilities = numpy.zeros(numpy.max(list(triad_dict.values())) + 1)
    for chief_edges in [((0, 0), (1, 2)),  # chief related connections have one of four orientations
                        ((0, 2), (1, 0)),
                        ((1, 0), (0, 2)),
                        ((1, 2), (0, 0))]:
        constr_mat = numpy.zeros((3, 3), dtype=bool)
        constr_mat[chief_edges] = True  # Set chief related edges to true
        # And for the rest iterate over the 16 possible patterns and identify their triad motif.
        probabilities += 0.25 * _count_possible_triads_constrained(constr_mat, p_rand)
    return probabilities


def expected_triad_probabilities_complex_control(subM):
    """
    expected_triad_probabilities_complex_control: The probabilities of each triadic motif in a control model that
    takes into account the chief-tribe sampling (i.e. that each non-chief must have some kind of connection to the
    chief), but is otherwise Erdos-Renyi for all other edges.
    :param subM: The adjacency matrix of the sampled tribe
    :return: The probability of the various triadic motifs, ordered as in Gal et al., 2017. Note: only connectected
    motifs are considered, i.e. motifs with less than 2 connections or only a single bidirectional connection are not
    considered. Therefore, the probabilities do not add up to 1.Calculates the probabilities
    """
    n_w_chief = comb(subM.shape[0] - 1, 2)  # number of triads that include the chief
    n_wo_chief = comb(subM.shape[0] - 1, 3)  # number of triads not including the chief
    n_chief_cons = subM.shape[0] - 1  # number of chief-related connections
    n_rand_cons = subM.sum() - n_chief_cons  # non-chief related connections
    ttl_pairs = subM.shape[0] * (subM.shape[0] - 1)  # number of potential connections
    p_rand = n_rand_cons / (ttl_pairs - n_chief_cons)  # prob. of non-chief related connections

    probabilities = n_w_chief * expected_triad_probabilities_w_chief(p_rand) + \
        n_wo_chief * expected_triad_probabilities_er(p_rand)
    return probabilities


def expected_triad_counts_simple_control(subM):
    """
    expected_triad_counts_simple_control: The expected number of each triadic motif in an Erdos-Renyi control model
    with the same connection probability as a given sample
    :param subM: The adjacency matrix of a neuron sample
    :return: The expected numbers of the various triadic motifs, ordered as in Gal et al., 2017. Note: only
    connectected motifs are considered, i.e. motifs with less than 2 connections or only a single bidirectional
    connection are not considered.
    """
    n_possible_triads = comb(subM.shape[0], 3)
    p = subM.sum() / (subM.shape[0] * (subM.shape[0] - 1))
    return n_possible_triads * expected_triad_probabilities_er(p)


def count_triads_of_chief(subM, chief_idx):
    """OBSOLETE. We use count_triads_fully_connected instead"""
    non_chiefs = numpy.setdiff1d(range(subM.shape[0]), chief_idx)
    counts = numpy.zeros(numpy.max(list(triad_dict.values())) + 1)
    for i, idx_one in enumerate(non_chiefs):
        for idx_two in non_chiefs[(i + 1):]:
            tstM = subM[:, [idx_one, idx_two, chief_idx]][[idx_one, idx_two, chief_idx]]
            counts[identify_motif(tstM)] += 1
    return counts


def count_triads_fully_connected(subM, max_num_sampled=5000000):
    """
    count_triads_fully_connected: Counts the numbers of each triadic motif in the sampled adjacency matrix
    :param subM: The adjacency matrix of a neuron sample
    :param max_num_sampled: (default: 5,000,000); The maximal number of connected triads classified. If the number of
    connected triads is higher than that, only the specified number is classified and the counts are extrapolated as
    actual_num_triads * counts / max_num_sampled
    :return: The counts of the various triadic motifs in the sample, ordered as in Gal et al., 2017. Note: only
    connectected motifs are counted, i.e. motifs with less than 2 connections or only a single bidirectional
    connection are not counted.
    """
    import time
    t0 = time.time()
    either_dirM = subM | subM.transpose()
    either_cn = numpy.dot(either_dirM, either_dirM)
    either_cn = numpy.triu(either_cn, 1)
    cn_idx = numpy.nonzero(either_cn)
    triads = set()
    print("Testing {0} potential triadic pairs".format(len(cn_idx[0])))
    for x, y in zip(*cn_idx):
        zs = numpy.nonzero(either_dirM[x] & either_dirM[y])[0]
        for z in zs:
            triads.add(tuple(sorted([x, y, z])))
    triads = list(triads)
    print("Time spent finding triads: {0}".format(time.time() - t0))
    print("Found {0} triads".format(len(triads)))
    t0 = time.time()
    counts = numpy.zeros(numpy.max(list(triad_dict.values())) + 1)
    smpl_idx = numpy.random.choice(len(triads),
                                   numpy.minimum(max_num_sampled, len(triads)),
                                   replace=False)
    for idx in smpl_idx:
        triad = triads[idx]
        motif_id = identify_motif(subM[:, triad][triad, :])
        counts[motif_id] += 1
    print("Time spent classifying triads: {0}".format(time.time() - t0))
    return (len(triads) * counts / len(smpl_idx)).astype(int)


def count_triads_all(tribes, M, info, cfg, **kwargs):
    """
    count_triads_all: The main analysis of this pipeline step. Counts the number of each type of triad motif in each
    sample and calculates their expected numbers in two control models
    :param tribes: TopoData; The tribes that were sampled in the sample_tribes-*.py steps
    :param M: scipe.sparse.csc; The adjacency matrix of the entire circuit
    :param info: pandas.DataFrame; basic information on all neurons in the circuit
    :param cfg: dict; the configuration of this pipeline step, read from common_config
    :param kwargs: Optional filtering of the neuron samples to be considered (e.g. index=0, sampling=M-type, etc.)
    :return: a ConditionCollection of 3x13 numpy.arrays, where: First row: ctual counts of triad motifs. Second row:
    expected counts according to a simple er model. Third row: expected counts according to a more complex model that
    takes the chief-tribe sampling into account (i.e. that there's at least one connection between the chief and each
    tribe member).
    """
    tribal_gids = tribes['gids']
    #  tribal_chiefs = tribes['chief']
    #  tribal_chiefs = tribal_chiefs.filter(**kwargs)
    tribal_gids = tribal_gids.filter(**kwargs)
    lst_results = []
    converter = GidConverter(info)
    for gid_res in tribal_gids.contents:
        print("Counting triads for: {0}".format(gid_res.cond))
        subM = submatrix(gid_res.res, M, converter)
        sampled = count_triads_fully_connected(subM, max_num_sampled=cfg.get("max_num_sampled", None))
        ctrl_smpl = expected_triad_counts_simple_control(subM)
        ctrl_compl = expected_triad_probabilities_complex_control(subM)
        res = numpy.vstack([sampled, ctrl_smpl, ctrl_compl])
        lst_results.append(ResultsWithConditions(res, **gid_res.cond))
    return ConditionCollection(lst_results)


def write_output(data, output_config):
    import os
    import json
    data = data.map(lambda x: {"overexpression": x.tolist()})
    data_dict = TopoData.condition_collection_to_dict(data)
    fn_out = output_config["triads"]

    if os.path.exists(fn_out):
        with open(fn_out, "r") as fid:
            existing = json.load(fid)
        for k, v in data_dict.items():
            existing.setdefault(k, {}).update(v)
        with open(fn_out, "w") as fid:
            json.dump(existing, fid, indent=2)
    else:
        with open(fn_out, 'w') as fid:
            json.dump(data_dict, fid, indent=2)


def main(path_to_config, **kwargs):
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("count_triads")
    tribes, M, info = read_input(stage["inputs"])
    overexpression = count_triads_all(tribes, M, info, stage['config'], **kwargs)
    write_output(overexpression, stage["outputs"])


def parse_filter_arguments(*args):
    fltr_dict = {}
    for arg in args:
        if "=" in arg:
            splt = arg.split("=")
            fltr_dict[splt[0]] = splt[1]
    return fltr_dict


if __name__ == "__main__":
    import sys
    main(sys.argv[1], **parse_filter_arguments(*sys.argv[2:]))
