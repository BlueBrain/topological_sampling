from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy
from toposample import config

def execute_classifier(features, stage_cfg):
	X = StandardScaler().fit_transform(features[:,:-1])
	y = features[:,-1]

	# dividing X, y into train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = stage_cfg["train_size"], random_state = 0)

	clf = SVC(cache_size = 500,gamma='scale')
	cv_scores = cross_val_score(clf, X, y, cv=4)
	test_scores = []

	for _ in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = stage_cfg["train_size"])
		test_scores.append(clf.fit(X_train,y_train).score(X_test,y_test))

    return test_scores, cv_scores

def read_input(input_config):
    features = numpy.load(input_config["features"])
    return features

def write_output(data, output_config):
    numpy.save(output_config["classifier_test_scores"], data[0])
    numpy.save(output_config["classifier_cv_scores"], data[1])

def main(path_to_config):
    cfg = config.Config(path_to_config)
    stage = cfg.stage("classifier")
    features = read_input(stage["inputs"])
    scores = execute_classifier(features, stage)
    write_output(scores, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
