import numpy as np
from nltk.classify.api import ClassifierI

class SklearnNLTKClassifier(ClassifierI):
    def __init__(self, classifier):
        self._classifier = classifier
        self._labels = None

    def classify(self, features):
        features_vector = self._dict_to_vector(features)
        return self._classifier.predict(features_vector)[0]

    def classify_many(self, featuresets):
        features_matrix = np.array([self._dict_to_vector(fs)[0] for fs in featuresets])
        return self._classifier.predict(features_matrix)

    def prob_classify(self, features):
        raise NotImplementedError("Probability estimation not implemented")

    def labels(self):
        if self._labels is None:
            self._labels = self._classifier.classes_
        return self._labels

    def _dict_to_vector(self, features_dict):
        return np.array([list(features_dict.values())])
