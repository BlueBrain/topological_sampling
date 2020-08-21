High level description: Generates the database of specified topological parameters for tribes centered around each of the neurons. Also adds info on the location, type and layer of those chiefs.

Language(s): Python

Additional dependencies: Pandas, networkx

Sub-steps: Collection of functions injecting into DataFrame, each function responsible for a single parameter. NOTE: creating the database from scratch with multiple different tribe parameters injected will take a long time, recommended to run separately once and store for future analyses.
		python pipeline/gen_topo_db/gen_topo_db.py working_dir/config/common_config.json

To speed things up, it is possible to build the database one parameter at a time. The name of the parameter is simply added as an additional argument. Calculations of individual parameters can thereby be parallelized:
        python pipeline/gen_topo_db/gen_topo_db.py working_dir/config/common_config.json "Euler characteristic"
        python pipeline/gen_topo_db/gen_topo_db.py working_dir/config/common_config.json "Betti numbers"
        ...
        python pipeline/gen_topo_db/gen_topo_db.py working_dir/config/common_config.json "tribe"
   This needs to be run for all parameters / columns listed under "parameters" in topo_db_config.json, plus "tribe" and "neuron_info"
   After that, merge the individual columns by running:
        python pipeline/gen_topo_db/merge_database.py working_dir/config/common_config.json
