import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

"""
Implement a classifier with required functions:

get_features: feature vector for each sample (1-hot, n-hot encodings or etc.)
fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels
"""


class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None
        self.vectorizer = None

    def get_features(self, text_list, ngram=1):
        """ Get word count features per sentences as a 2D numpy
        array as values and tweetids as keys"""
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(ngram_range=(ngram, ngram),
                                              stop_words='english')
            vector = self.vectorizer.fit_transform(text_list)
        else:
            vector = self.vectorizer.transform(text_list)

        features_array = vector.toarray()
        return features_array

    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        pass
