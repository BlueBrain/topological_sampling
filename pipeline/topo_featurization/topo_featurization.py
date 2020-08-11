import numpy as np
from toposample import config
import json


def read_input(input_config):
    spiketrains = np.load(input_config["split_spikes"])
    tribes = json.load(input_config["tribes"])
    return spiketrains,tribes

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
    spiketrains,tribes = read_input(stage["inputs"]) # TODO: check the format of tribes.json

    topo_featurization_cfg = stage["config"]["topological_featurization"]

    timebin = topo_featurization_cfg["time_bin"]

    # tribes_for_featurization contains active subtribes in each time bin: first n positions are the n active subtribes for the first timebin,
    # in descending order by their chiefs. Then follows the n active subtribes for the second timebin etc.
    for stimulus_class in range(len(spiketrains)):
        experiment_ID = 0
        
        for experiment in spiketrains[stimulus_class]:
            tribes_for_featurization = []

            for t in range(20): # TODO: not hard-coded to 20, depends on stimulation experiment duration
                spikers = (experiment[(experiment[:,0] > t*timebin) & (experiment[:,0] <= (t+1)*timebin)][:,1]-62693).astype(dtype=int) # TODO: check the need for the magical number 62693

                for tribe in tribes: 
                    tribes_for_featurization.append(np.intersect1d(tribe,spikers))

            if topo_featurization_cfg["topo_method"][0] == "EC":
                # get Euler characteristic of the active subcommunities
                feature_vector = [Flagsering(parameter,tribe) for tribe in tribes_for_featurization] # TODO: incorporate Flagser and Flagsering utilities, make separate module?
                feature_vector.append(stimulus_class)
                feature_vectors.append(feature_vector)
                print(f'{parameter} Activation class {stimulus_class} Experiment {experiment_ID} featurized by {topo_featurization_cfg["topo_method"][1]}')
                experiment_ID += 1

    write_output(np.array(feature_vectors), stage["outputs"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1])