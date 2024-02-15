import numpy as np
import scipy
from abs_custom_classifier_with_feature_generator import CustomClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Implement a KNN classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels
"""


class CustomKNN(CustomClassifier):
    def __init__(self, k=5, distance_metric='cosine'):
        """ """
        super().__init__()

        self.k = k
        self.train_feats = None
        self.train_labels = None
        self.is_trained = False
        self.distance_metric = distance_metric

    def fit(self, train_feats, train_labels):
        """ Fit training data for classifier """

        self.train_feats = train_feats
        self.train_labels = np.array(train_labels)

        self.is_trained = True
        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """

        assert self.is_trained, 'Model must be trained before predicting'

        # 2D array of distances between all test and all training samples
        # Shape (Test X Train)
        predictions = []

        distance_values = scipy.spatial.distance.cdist(
            test_feats, self.train_feats, 'euclidean')
        for test in distance_values:
            distance_values_labels = [(test[i], self.train_labels[i])
                                      for i in range(len(test))]
            labels = [dist_labels[1] for dist_labels
                      in sorted(distance_values_labels)[:self.k]]
            predictions.append(max(labels))

        return predictions
