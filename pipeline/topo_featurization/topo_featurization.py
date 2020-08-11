import numpy as np
import pandas as pd
import pickle as pkl
import importlib
from toposample import config
import importlib

def read_input(input_config):
    spiketrains = np.load(input_config["split_spikes"])
    return spiketrains

def write_output(feature_vectors,output_config):
	np.save(feature_vectors,output_config["features"])

def main(path_to_config):
	# Empty list to hold feature vectors
	feature_vectors = []

    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("topological_featurization")

    # Fetch spiketrains
    spiketrains = read_input(stage["inputs"])

    topo_featurization_cfg = stage["config"]["topological_featurization"]

    timebin = topo_featurization_cfg["time_bin"]

    # communities_for_featurization contains active subcommunities in each time bin: first n positions are the n active subcommunities for the first timebin,
    # in descending order by the structural parameter. Then follows the n active subcommunities for the second timebin etc.
    for activation_class in range(len(spiketrains)):
        experiment_ID = 0
        
        for experiment in spiketrains[activation_class]:
            communities_for_featurization = []

            for t in range(20): # This loop works OK
                spikers = (experiment[(experiment[:,0] > t*timebin) & (experiment[:,0] <= (t+1)*timebin)][:,1]-62693).astype(dtype=int)

                for community in communities:
                    communities_for_featurization.append(np.intersect1d(community,spikers))

            # get Euler characteristic of the active subcommunities
            feature_vector = [Flagsering(parameter,community) for community in communities_for_featurization]
            feature_vector.append(activation_class)
            feature_vectors.append(feature_vector)
            #print(f'{parameter} Activation class {activation_class} Experiment {experiment_ID}')
            output.write(f'{parameter} Activation class {activation_class} Experiment {experiment_ID}\n')
            experiment_ID += 1

    write_output(np.array(feature_vectors), stage["outputs"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1])