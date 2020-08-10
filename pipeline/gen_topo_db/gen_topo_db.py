import numpy as np
import pandas as pd
import pickle as pkl
import importlib
from toposample import config
import importlib

def read_input(input_config):
    adj_matrix = np.load(input_config["adjacency_matrix"])
    neuron_info = pkl.load(input_config["neuron_info"])
    return adj_matrix, neuron_info

def write_output(DB,output_config):
	DB.to_pickle(output_config["database"])

def add_tribes(adj_matrix):
	tribes = []

	for neuron in range(len(adj_matrix)):
		# Din are the in-neighbours of a vertex.
		# Dout are the out-neighbours of a vertex.
		Din = np.nonzero(adj_matrix[:,neuron])[0]
		Dout = np.nonzero(adj_matrix[neuron])[0]
		tribes.append(np.append(np.union1d(Din,Dout),neuron))

	return tribes

def main(path_to_config):
	# Initialize empty database
	DB = pd.DataFrame()

    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("gen_topo_db")

    # Fetch adjacency matrix and neuron infos according to configured locations/files
    adj_matrix, neuron_info = read_input(stage["inputs"])
    topo_db_cfg = stage["config"]["gen_topo_db"]
    precision = topo_db_cfg["precision"]

    # Compute all the tribes
    DB["tribe"] = add_tribes(adj_matrix)

    # Add actual GIDs
    DB["gid"] = DB.index + 62693

    # TODO: Add neuron info
    # add_neuron_info(neuron_info)

    # Loop over the tribe parameters as listed in topo_db_config, each loop makes a call to the associated function computing the parameter values and injects into DB.
    for parameter in topo_db_cfg["parameters"]:
    	module = importlib.import_module("../pipeline/gen_topo_db/" + topo_db_cfg[parameter]["source"])
    	DB[topo_db_cfg[parameter]["column_name"]] = module.compute(DB["tribe"],adj_matrix,precision)

    write_output(DB, stage["outputs"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1])