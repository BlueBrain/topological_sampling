# STAGE 19: Classifier

High level description: Runs a support vector machine on the feature vectors and saves the classification accuracies.

Language(s): Python

Additional dependencies: numpy, sklearn


Sub-steps:
    Run separately for classification based on topological features and based on manifol analysis:
        python pipeline/classifier/classifier.py working_dir/config/common_config.json features
        python pipeline/classifier/classifier.py working_dir/config/common_config.json manifold
    Alternatively, the two sub-steps also can be run in separate steps.
    This allows you to trivially parallelize the calculation:
        python pipeline/classifier/classifier.py working_dir/config/common_config.json features “sampling=M-type” “specifier=L23_PC”
        python pipeline/classifier/classifier.py working_dir/config/common_config.json features “sampling=M-type” “specifier=L4_PC”
        …
        python pipeline/classifier/classifier.py working_dir/config/common_config.json features “sampling=Parameter” “specifier=Betti 2”
    which is then repeated for classification based on the manifold analysis:
        python pipeline/classifier/classifier.py working_dir/config/common_config.json manifold “sampling=M-type” “specifier=L23_PC”
        python pipeline/classifier/classifier.py working_dir/config/common_config.json manifold “sampling=M-type” “specifier=L4_PC”
        …
        python pipeline/classifier/classifier.py working_dir/config/common_config.json manifold “sampling=Parameter” “specifier=Betti 2”

