# STAGE 10: Structural tribe analysis

High level description: Extracts the specified topological parameter values for all neuron samples. For tribal samples just uses the one of its chief, otherwise performs a prediction by tribal overlap size
Language(s): python
Additional python dependencies:
    Nonse
Sub-steps:
    1. lookup topological topological parameters for tribal samples from the topo_db
    2. predict a synthetic value of the topological parameters for volumetric samples.
    This is a weighted average of the parameter values for the tribes, where the weights are proportional to the size of the overlap of the tribe and the volumetric sample
    3. calculate actual values of topological parameters for the volumetric samples from scratch. (Only for a subset of parameters)

For sub-steps 1 and 2 simply run:
    python pipeline/struc_tribe_analysis/parameters_for_tribes.py working_dir/config/common_config.json
For sub-step 3 run:
    python pipeline/struc_tribe_analysis/topo_for_volumetric_samples.py
