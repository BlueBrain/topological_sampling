import numpy as np
import pandas as pd
import pickle as pkl
from toposample import config

def read_input(input_config):
    adj_matrix = np.load(input_config["adjacency_matrix"])
    neuron_info = pkl.load(input_config["neuron_info"])
    return adj_matrix, neuron_info

def write_output(DB,output_config):
	DB.to_pickle(output_config["database"])

def main(path_to_config):
	# Initialize empty database
	DB = pd.DataFrame()
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("gen_topo_db")
    # Fetch adjacency matrix and neuron infos according to configured locations/files
    adj_matrix, neuron_info = read_input(stage["inputs"])
    write_output(DB, stage["outputs"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1])