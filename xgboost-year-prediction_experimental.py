# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment.

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import os
import time
from pandas import read_csv

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt

import pickle


best_params = []

# Auxillary Function for getting model name througout validation
def get_model_name():
    return 'saved_models/model_' + '.h5'


def main(filename, test):
    print("GETTING DATASET")

    data = read_csv('data_library/MSD_unbiased_300.csv', header=None, delimiter=",")
    dataset = data.values
    # split data into X and y
    training_examples = dataset[:, 1:89]
    training_labels = dataset[:, 0]
    print("SPLITTING TRAINING AND VALIDATION SETS")




    for x in range(len(training_labels)):
        if int(training_labels[x]) < 1930:
            training_labels[x] = 0
        elif int(training_labels[x]) < 1940 and int(training_labels[x]) >= 1930:
            training_labels[x] = 1
        elif int(training_labels[x]) < 1950 and int(training_labels[x]) >= 1940:
            training_labels[x] = 2
        elif int(training_labels[x]) < 1960 and int(training_labels[x]) >= 1950:
            training_labels[x] = 3
        elif int(training_labels[x]) < 1970 and int(training_labels[x]) >= 1960:
            training_labels[x] = 4
        elif int(training_labels[x]) < 1980 and int(training_labels[x]) >= 1970:
            training_labels[x] = 5
        elif int(training_labels[x]) < 1990 and int(training_labels[x]) >= 1930:
            training_labels[x] = 6
        elif int(training_labels[x]) < 2000 and int(training_labels[x]) >= 1930:
            training_labels[x] = 7
        else:
            training_labels[x] = 8



    seed = 7
    X = training_examples
    Y = training_labels
    label_encoded_y = LabelEncoder().fit_transform(Y)

    # fit model no training data
    model = XGBClassifier()

    if test is False:

        # encode string class values as integers

        n_estimators = [50, 100, 150, 200]
        max_depth = [2, 4, 6, 8]

        param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
        grid_result = grid_search.fit(X, label_encoded_y)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_params = list(grid_result.best_params_.values())
        best_params = dict(max_depth=[best_params[0]], n_estimators=[best_params[1]])

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        # plot
        scores = np.array(means).reshape(len(max_depth), len(n_estimators))
        for i, value in enumerate(max_depth):
            plt.plot(n_estimators, scores[i], label='depth: ' + str(value))


        plt.errorbar(n_estimators, means, yerr=stds)
        plt.title("XGBoost n_estimators vs Log Loss")
        plt.xlabel('n_estimators')
        plt.ylabel('Log Loss')
        plt.savefig('plots/n_estimators.png')

        # Recap with best model
        model = XGBClassifier()
        GridSearchCV(model, best_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
        model = grid_search.fit(X, label_encoded_y)
        pickle.dump(model, open("saved_models/XGB.pickle.dat", 'wb'))


    else:
        model = pickle.load(open("saved_models/XGB.pickle.dat", "rb"))

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.9, random_state=7)

        model.fit(X_train, y_train)

        # make predictions for test data
        predictions = model.predict(X_test)
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("TEST ACCURACY: %.2f%%" % (accuracy * 100.0))

        print(predictions)



if __name__ == "__main__":

    main(filename="data_library/MSD_unbiased_300.csv", test=False)
    plt.show()
    print("\n INITIALIZE TESTING")
    main(filename="data_library/MSDB_5000_testset.csv", test=True)

