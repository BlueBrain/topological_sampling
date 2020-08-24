# topological_sampling
The analysis pipeline for the topological sampling project

First install the python dependencies for reading / writing results and configurations:
- cd common/toposample_utilities
- pip install .

Next, install the python dependencies for counting directed simplex containment:
- cd common/pyflagsercontain
- pip install .

Next, obtain the input data used for our paper (or use your own simulation results!) under the following link: LINK HERE and place it into working_dir/data/input_data.

Then run the pipeline steps in order:
- pipeline/gen_topo_db
- pipeline/sample_tribes
- pipeline/struc_tribe_analysis
- pipeline/count_triads
- pipeline/split_time_windows
- pipeline/manifold_analysis
- pipeline/topological_featurization
- pipeline/classifier

Each step of the pipelin provides its own README that describes how to run it. It also lists potential additional dependencies in addition to the ones listed in common/setup.py

Configuration of each step and the pipeline as a whole is given in working_dir/config. If you want to change some of the parameters either edit them directly, or create a copy of the working_dir and change the configuration in that copy and work there.

After running the pipeline the analyzed output data is found under working_dir/data/analyzed_data.

jupyter notebooks that use the analyzed data to create the figures (or individual panels) of the paper can be found under notebooks:
- cd notebooks
- jupyter notebook

They create the figures and also provide a good starting point for learning how to work with the input and/or analyzed data.

