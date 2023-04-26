import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import pickle
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')
SVM_CLASSIFIER = 'svm_classifier.sav'

DECISION_TREE_CLASSIFIER = 'decision_tree_classifier.sav'

GRADIENT_BOOSTING_CLASSIFIER = 'gradient_boosting_classifier.sav'

RANDOM_FOREST_CLASSIFIER = 'random_forest_classifier.sav'

MODEL_FOLDER = './grid_search/'


# learn a random forest regressor model from the train set and apply it on the test set
def random_forest_classification(X_train, y_train, filename):
    filename = MODEL_FOLDER + filename + RANDOM_FOREST_CLASSIFIER
    try:
        classifier = pickle.load(open(filename, "rb"))
    except (OSError, IOError) as e:
        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(X_train, y_train)
        pickle.dump(classifier, open(filename, 'wb'))

    return classifier


def svm_classification(X_train, y_train, filename):
    filename = MODEL_FOLDER + filename + SVM_CLASSIFIER
    try:
        classifier = pickle.load(open(filename, "rb"))
    except (OSError, IOError) as e:
        classifier = svm.SVC(decision_function_shape='ovo')
        classifier.fit(X_train, y_train)
        pickle.dump(classifier, open(filename, 'wb'))

    return classifier


def gradient_boosting_classification(X_train, y_train, filename):
    filename = MODEL_FOLDER + filename + GRADIENT_BOOSTING_CLASSIFIER
    try:
        classifier = pickle.load(open(filename, "rb"))
    except (OSError, IOError) as e:
        classifier = GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=0)
        classifier.fit(X_train, y_train)
        pickle.dump(classifier, open(filename, 'wb'))

    return classifier


def decision_tree_classification(X_train, y_train, filename):
    filename = MODEL_FOLDER + filename + DECISION_TREE_CLASSIFIER

    try:
        classifier = pickle.load(open(filename, "rb"))
    except (OSError, IOError) as e:
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        pickle.dump(classifier, open(filename, 'wb'))

    return classifier


def save_model(classifier, name):
    pickle.dump(classifier, open(name, 'wb'))


def get_random_best_estimator(classifier, X_train, y_train, random_grid):
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=100, cv=3,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    pprint(rf_random.best_params_)

    return rf_random


def get_grid_best_estimator(classifier, X_train, y_train, param_grid):
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                               cv=3, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    pprint(grid_search.best_params_)

    return grid_search


def evaluate_results(classifier, X_test, y_test, result_file=None, n_classes=2):
    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))
    if result_file is not None:
        result_file.write('\n' + str(classification_report(y_test, y_pred)))
    # print('Multi-label confusion matrix: ')
    # print(multilabel_confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_array = np.array(cm)

    group_counts = ["{0:0.0f}".format(value) for value in
                    cm_array.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm_array.flatten() / np.sum(cm_array)]
    labels = [f"{v2}\n{v3}" for v2, v3 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(n_classes, n_classes)
    sns.heatmap(cm_array, annot=labels, fmt='', cmap='Blues')
    model_name = type(classifier).__name__
    plt.ylabel('Actual Classes', fontweight='bold')
    plt.xlabel('Predicted Classes', fontweight='bold')

    moment = time.strftime("%H_%M_%S", time.localtime())
    plt.savefig(f"./final_results/cm_{model_name}_{moment}.eps", dpi=400)
    plt.show()


def apply_grid_search_on_models():
    file1.write(f"\nRandom Forest Classifier")
    best_grid = get_grid_best_estimator(rf_classifier, X_train_transformed, y_train, random_forest_grid)
    file1.write('\n' + str(best_grid.best_params_))
    evaluate_results(best_grid.best_estimator_, X_test_transformed, y_test, file1, n_classes=2)
    moment = time.strftime("%H_%M_%S", time.localtime())
    save_model(best_grid.best_estimator_,
               MODEL_FOLDER + f'{course}_{num_weeks}_weeks_{moment}_' + RANDOM_FOREST_CLASSIFIER)

    file1.write(f"\nGradient Boosting Classifier")
    best_grid = get_grid_best_estimator(gb_classifier, X_train_transformed, y_train, gb_grid)
    file1.write('\n' + str(best_grid.best_params_))
    evaluate_results(best_grid.best_estimator_, X_test_transformed, y_test, file1, n_classes=2)
    moment = time.strftime("%H_%M_%S", time.localtime())
    save_model(best_grid.best_estimator_,
               MODEL_FOLDER + f'{course}_{num_weeks}_weeks_{moment}_' + GRADIENT_BOOSTING_CLASSIFIER)

    file1.write(f"\nDecision Tree Classifier")
    best_grid = get_grid_best_estimator(decision_tree_classifier, X_train_transformed, y_train, decision_tree_grid)
    file1.write('\n' + str(best_grid.best_params_))
    evaluate_results(best_grid.best_estimator_, X_test_transformed, y_test, file1, n_classes=2)
    moment = time.strftime("%H_%M_%S", time.localtime())
    save_model(best_grid.best_estimator_,
               MODEL_FOLDER + f'{course}_{num_weeks}_weeks_{moment}_' + DECISION_TREE_CLASSIFIER)

    file1.write(f"\nSVM Classifier")
    best_random = get_random_best_estimator(svm_classifier, X_train_transformed, y_train, svm_grid)
    file1.write('\n' + str(best_random.best_params_))
    evaluate_results(best_random.best_estimator_, X_test_transformed, y_test, file1, n_classes=2)
    save_model(best_random.best_estimator_, MODEL_FOLDER + f'{course}_{num_weeks}_weeks_' + SVM_CLASSIFIER)


df = pd.read_csv('./data/clickstream_dataset_updated.csv')
df.columns = range(df.columns.size)
df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)
courses = pd.unique(df['course_id'])
demo_length = 37
weeks = [5, 10, 15, 20]


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = [None, 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_forest_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}

# List of C values
C_range = np.logspace(-1, 1, 3)
# List of gamma values
gamma_range = np.logspace(-1, 1, 3)
# Define the search space
svm_grid = {
    # Regularization parameter.
    "C": C_range,
    # Kernel type
    "kernel": ['rbf', 'poly'],
    # Gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    "gamma": gamma_range.tolist() + ['scale', 'auto']
}
gb_grid = {
    "n_estimators": n_estimators,
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    'max_depth': max_depth,
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1],
}

decision_tree_grid = {"max_depth": max_depth,
                      "max_features": max_features,
                      "min_samples_leaf": min_samples_leaf,
                      "criterion": ["gini", "entropy"]}
# pprint(random_forest_grid)

file1 = open("./final_results/grid_search_results.txt", "w")
num_classes = 2
for course in courses:

    train_df = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/train_dataset.csv')
    train_df.columns = range(train_df.columns.size)
    test_df = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/test_dataset.csv')
    test_df.columns = range(test_df.columns.size)

    if course not in ['AAA', 'CCC']:
        dev_df = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/dev_dataset.csv')
        dev_df.columns = range(dev_df.columns.size)

        train_df = pd.concat([train_df, dev_df], ignore_index=True)
    train_demo = train_df.iloc[:, -1 * (demo_length + 1):]

    test_demo = test_df.iloc[:, -1 * (demo_length + 1):]
    for num_weeks in weeks:
        print(f'Predicting for course {course} with {num_weeks} weeks.')
        # file1.write(f"Predicting for course {course} with {num_weeks} weeks.")
        X_train = train_df.iloc[:, :num_weeks * 20]
        X_train = pd.concat([X_train, train_demo], axis=1, ignore_index=True)
        X_train = X_train.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1:].values.ravel()

        X_test = test_df.iloc[:, :num_weeks * 20]
        X_test = pd.concat([X_test, test_demo], axis=1, ignore_index=True)
        X_test = X_test.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1:].values.ravel()

        rf_classifier = RandomForestClassifier()
        svm_classifier = svm.SVC()
        decision_tree_classifier = tree.DecisionTreeClassifier()
        gb_classifier = GradientBoostingClassifier()

        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        print('RF Classifier')
        evaluate_results(rf_classifier, X_test_transformed, y_test, None, n_classes=2)
        svm_classifier.fit(X_train_transformed, y_train)

        print('SVM Classifier')
        evaluate_results(svm_classifier, X_test_transformed, y_test, None, n_classes=2)
        decision_tree_classifier.fit(X_train_transformed, y_train)

        print('DT Classifier')
        evaluate_results(decision_tree_classifier, X_test_transformed, y_test, None, n_classes=2)

        print('GB Classifier')
        gb_classifier.fit(X_train_transformed, y_train)
        evaluate_results(gb_classifier, X_test_transformed, y_test, None, n_classes=2)

        #### This part is for applying grid search
        # apply_grid_search_on_models()

file1.close()
