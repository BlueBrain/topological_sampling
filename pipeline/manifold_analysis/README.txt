# STAGE 13: Manifold analysis

High level description: Transforms the spikes trains of neurons in a sample, extracting the ‘hidden components’
Language(s): python
Additional dependencies: sklearn

Sub-steps:
        python pipeline/manifold_analysis/manifold_analysis.py working_dir/config/common_config.json
    Alternatively can be run in separate steps. This allows you to trivially parallelize the calculation:
        python pipeline/manifold_analysis/manifold_analysis.py working_dir/config/common_config.json “sampling=M-type” “specifier=L23_PC”
        python pipeline/manifold_analysis/manifold_analysis.py working_dir/config/common_config.json “sampling=M-type” “specifier=L4_PC”
        …
        python pipeline/manifold_analysis/manifold_analysis.py working_dir/config/common_config.json “sampling=Parameter” “specifier=Betti 2”
