High level description: Extract from spike train data active subtribes in each time bin as specified in config. For all of these compute topological descriptor, such as EC. Concatenate with experiment identifier to output feature vectors for every activity experiment.

Language(s): Python

Sub-steps: 
		Loop over all split spiketrains and input tribes. Intersect spiking neurons in time bins with tribes and put these into intermediate list.
		Map the list to a list of corresponding topological descriptors by calling selected topological method as specified in config.
		python pipeline/topo_featurization/topo_featurization.py working_dir/config/common_config.json
