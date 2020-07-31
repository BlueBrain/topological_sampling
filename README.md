# topological_sampling
The analysis pipeline for the topological sampling project

First install the python dependencies:
cd common
pip install .

Then run the pipeline steps in order:
pipeline/gen_topo_db
pipeline/sample_tribes
pipeline/struc_tribe_analysis
pipeline/split_time_windows
pipeline/manifold_analysis
pipeline/topological_featurization
pipeline/classifier

Each step of the pipelin provides its own README that describes how to run it.
Configuration of each step and the pipeline as a whole is given in working_dir/config. If you want to change some of the parameters it is best to create a copy of the working_dir, change the configuration in that copy and work there.

The input data used for our paper can be obtained at: LINK HERE.
The data is to be put into working_dir/data/input_data
After running the pipeline the analyzed output data is found under working_dir/data/analyzed_data.

jupyter notebooks that use the analyzed data to create the figures (or individual panels) of the paper can be found under notebooks:
cd notebooks
jupyter notebook

They create the figures and also provide a good starting point for learning how to work with the input and/or analyzed data.

