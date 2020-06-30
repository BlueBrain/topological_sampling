# STAGE 3c: Split time windows

High level description: Takes the raw spikes and stimulus information and reassembles the spikes input one
list per stimulus as specified in 3d. Also subtracts the time of the start of the stimulus presentation,
such that all spike times are relative to the beginning of the current stimulus presentation
Language(s): Python
Sub-steps:
After pip installing the "toposample" package, simply run:

python pipeline/split_time_windows/run_split_time_windows.py working_dir/config/common_config.json
