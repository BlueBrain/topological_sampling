High level description: Generates the database of specified topological parameters for tribes centered around each of the neurons. Also adds info on the location, type and layer of those chiefs.

Language(s): Python

Additional dependencies: Pandas, networkx

Sub-steps: Collection of functions injecting into DataFrame, each function responsible for a single parameter. NOTE: creating the database from scratch with multiple different tribe parameters injected will take a long time, recommended to run separately once and store for future analyses.
		python pipeline/gen_topo_db/gen_topo_db.py working_dir/config/common_config.json