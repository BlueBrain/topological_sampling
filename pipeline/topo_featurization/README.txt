# STAGE X: Topological featurization

Extracts the time series of a specified topological parameter for the "active" sub-tribes of all tribes.
The "active" subtribe is the subgraph of containing only the neurons of a tribe that are spiking in a give time step.
Language(s): python
Additional python dependencies:
    pyflagser (https://github.com/giotto-ai/pyflagser or just pip install pyflagser)
Sub-steps:
        Loop over all split spiketrains and input tribes. Intersect spiking neurons in time bins with tribes and put these into intermediate list.
		Map the list to a list of corresponding topological descriptors by calling selected topological method as specified in config.
To run:
    python pipeline/topo_featurization/topo_featurization.py working_dir/config/common_config.json
However, this step is quite costly. It is recommended to run this embarrassingly parallel by executing the analysis
for each sampling / specifier combination separately:
    python pipeline/topo_featurization/topo_featurization.py working_dir/config/common_config.json “sampling=M-type” “specifier=L23_PC”
    python pipeline/topo_featurization/topo_featurization.py working_dir/config/common_config.json “sampling=M-type” “specifier=L4_PC”
        …
    python pipeline/topo_featurization/topo_featurization.py working_dir/config/common_config.json “sampling=Parameter” “specifier=Betti 2”

