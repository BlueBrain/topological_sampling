# NEUTRINO - NEUral TRIbe and Network Observer
An analysis pipeline for joint analysis of neural spiking data and synaptic connectivity.
This repository encompasses all analyses performed in the following manuscript:

MANUSCRIPT HERE

## Requirements
- cmake >= 3.4
- python >= 3.7
  - h5py
  - numpy
  - simplejson
  - scipy >= 1.0.0
  - scikit-learn
  - pandas
  - progressbar
  - pyflagser
  - future
  - networkx
  
## Installation

Download the repository and submodules with:
- git clone --recursive https://github.com/MWolfR/topological_sampling.git

Then, install the python dependencies for reading / writing results and configurations:
- cd common/toposample_utilities
- pip install .

Next, install the python dependencies for counting directed simplex containment:
- cd common/pyflagsercontain
- pip install .

## Usage

Next, obtain the input data used for our paper (or use your own simulation results!) under the following link: LINK HERE and place it into working_dir/data/input_data.

The following four input files are required, the addresses for which should be updated in working_dir/config/common_config.json:
- The adjacency matrix of the circuit in scipy sparse csr format
- The spike trains of the simulation
- The classification labels for the the stimuli as a numpy array
- A pandas database containing the layers, morphilogical type and (x,y,z) coordinates of the neurons

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

## Citation

If you decide to use any part of this software, please reference the following manuscript:

MANUSCRIPT HERE
