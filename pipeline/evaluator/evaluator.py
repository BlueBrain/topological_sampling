import numpy
from toposample import config

def write_output(scores, stage):
    with open(stage["output"],'w') as f:
	    f.write('CV accuracy: %0.2f +/- %0.2f,  test accuracy: %0.2f +/- %0.2f \n' % (scores[0].mean(),scores[0].std()*2,np.array(scores[1]).mean(),np.array(scores[1]).std()*2))


def read_input(input_config):
    test_scores = numpy.load(input_config["test_scores"])
    cv_scores = numpy.load(input_config["cv_scores"])
    return test_scores, cv_scores


def main(path_to_config):
    cfg = config.Config(path_to_config)
    stage = cfg.stage("evaluator")
    scores = read_input(stage["inputs"])
    write_output(scores, stage)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
