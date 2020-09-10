# STAGE 10: Structural tribe analysis

High level description: Extracts the specified topological parameter values for all neuron samples. For tribal samples just uses the one of its chief, otherwise performs a prediction by tribal overlap size
Language(s): python
Additional python dependencies:
    Nonse
Sub-steps:
    0. (Optional): Calculate the optimal number of tribes to include in the weighted average in step 2.
    This fills in optimized values into the configuration file for this stage. Alternatively you can manually fill them in.
    The default value in the configuration file in the code repo is already optimized.
    1. lookup topological topological parameters for tribal samples from the topo_db
    2. predict a synthetic value of the topological parameters for volumetric samples.
    This is a weighted average of the parameter values for the tribes, where the weights are proportional to the size of the overlap of the tribe and the volumetric sample
    3. calculate actual values of topological parameters for the volumetric samples from scratch. (Only for a subset of parameters)

For sub-step 0 run:
    python pipeline/struc_tribe_analysis/best_predictor_for_volumetric.py working_dir/config/common_config.json
       or:
    python pipeline/struc_tribe_analysis/best_predictor_for_volumetric.py -o output_fn working_dir/config/common_config.json
       to additionally generate a plot of the optimization at the specified output_fn
For sub-steps 1 and 2 simply run:
    python pipeline/struc_tribe_analysis/parameters_for_tribes.py working_dir/config/common_config.json
For sub-step 3 run:
    python pipeline/struc_tribe_analysis/topo_for_volumetric_samples.py
