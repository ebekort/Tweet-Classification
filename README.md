# Classification Models Analysis on whether tweets are factcheckable or not

This github consists of python scripts that analyse tweets if they can be factchecked or not. Using supervised learning, meaning that tweets have been labelled, classifiers are trained and new tweets that the models have not been seen before can be classified.

## Files

### `assignment_classification.py`

This is the main script that brings all the individual parts together. In this script tweets are read and preprocessed, classifiers are trained and tweets are classified with the help of other scripts in this github, results are evaluated with and without cross-validation.

**Functions:**
read_dataset() reads data from a folder and splits it into a list
preprocess_dataset(text_list) preprocesses text lists, it does this by removing certain special characters, urls, hashtags, usertags, numbers and it converts everything to lower case
evaluate(true_labels, predict_labels) evaluates the predicted labels with the true labels and calculates accuracy, precision, recall and f1 score
train_test(train_data, train_labels, test_data, test_labels, classifier) trains a classifier (svm, Naive bayes or knn) and evaluates it's performance
cross_validate(train_data, train_labels, n_fold=10, classifier='svm') does a n-fold cross validation to split up the data when their is no test data available
main() takes everything together and makes sure everything is executed in the right order

### `abs_custom_classifier_with_feature_generator.py`

This script makes a class CustomClassifier that includes functions that are common among most classifiers so they can inheritated

### `custom_knn_classifier.py`

this script makes a CustomKNN class that uses a custom made k-nearest neighbors approach and inherited basic functions of CustomClassifier

### `custom_naive_bayes.py`

This script implements a custom Naive Bayes classifier by extending `CustomClassifier`.
this script makes a CustomNaiveBayes class that uses a custom made naive bayes approach and also inherited basic functions of CustomClassifier

## Usage

To use one of the custom made classification models in this github the following steps must be taken:

prepare the data: use .tsv files must be used. each row represents a tweets with at least 2 columns: tweet_text, class_label
data used by my research has been provided in the github repository too in the CT22_dutch_1B_claim folder

run the classifier: python ./assignment_classification.py classifiermethod cross_validate
classifiermethod can either be svm (support vector model), naive_bayes (naive bayes), knn (k-nearest neighbours)
cross_validate can be added to use cross validation or removed to use seperate test_data, doing that a seperate file containing test data must be provided

results will be printed automatically by the system: accuracy, precision, recall, f1 score

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `scipy`

All these libraries must be installed in order for this program to run.

## Contributors

- [Ebe Kort](https://github.com/ebekort)
