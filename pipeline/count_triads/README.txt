# STAGE 10c: Triad counts

High level description: Counts the number of each type of triad motif in the neuron samples. Also calculates expected numbers according to two control models.
The first control model is simply Erdos-Renyi; the second one takes into account the chief-tribe type of sampling. The code will use the
second type of control model also for volumetric samples, but those results are arguably meaningless.
Language(s): python
Additional dependencies: pandas

Sub-steps:
        python pipeline/count_triads/count_triads.py working_dir/config/common_config.json
    Alternatively can be run in separate steps. This allows you to trivially parallelize the calculation:
        python pipeline/count_triads/count_triads.py working_dir/config/common_config.json "sampling=M-type" "specifier=L23_PC"
        python pipeline/count_triads/count_triads.py working_dir/config/common_config.json "sampling=M-type" "specifier=L4_PC"
        …
        python pipeline/count_triads/count_triads.py working_dir/config/common_config.json "sampling=Parameter" "specifier=Betti 2"
