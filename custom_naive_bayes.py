import numpy as np
from abs_custom_classifier_with_feature_generator import CustomClassifier
from sklearn.naive_bayes import ComplementNB


"""
Implement a Naive Bayes classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels
"""


class CustomNaiveBayes(CustomClassifier):

    def __init__(self, alpha=1.0):
        """ """
        super().__init__()

        self.alpha = alpha
        self.prior = None
        self.classifier = None

    def fit(self, train_feats, train_labels):
        """ Calculate the priors, fit training data for
        Naive Bayes classifier """

        '''for sample in train_feats[:20]:
            for s in sample:
                if s != 0:
                    print(s)'''

        '''
        own implementation of naive bayes, however not able to use
        predict function since the model does not recognize it as
        being trained

        checkable = []
        uncheckable = []
        for i in range(len(train_labels)):
            if train_labels[i] == 1:
                checkable.append(train_feats[i])
            else:
                uncheckable.append(train_feats[i])

        lencheck = len(checkable)
        lenuncheck = len(uncheckable)
        checkable = np.array(checkable).sum(axis=0)
        uncheckable = np.array(uncheckable).sum(axis=0)
        totalcheck = checkable.sum()
        totaluncheck = uncheckable.sum()
        checkable = checkable/totalcheck
        uncheckable = uncheckable/totaluncheck
        p (terrible|helpful) * p(product|helpful) * p(helpful)
        print(lencheck+lenuncheck)
        print(len(train_labels))
        Pcheckable = lencheck/(lencheck+lenuncheck)
        Puncheckable = lenuncheck/(lencheck+lenuncheck)'''

        self.mnb = ComplementNB()
        self.mnb.fit(train_feats, train_labels)

        self.classifier = []

        self.is_trained = True
        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """

        assert self.is_trained, 'Model must be trained before predicting'

        # Use the scikit-learn predict function
        predictions = self.mnb.predict(test_feats)
        return predictions
