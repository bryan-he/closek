#!/usr/bin/env python

"""Basic test running close-k and baselines on monk3."""

import numpy as np
import sklearn.linear_model
import pmlb

import closek

if __name__ == "__main__":
    np.random.seed(0)

    # Load dataset
    X, y = pmlb.fetch_data("monk3", return_X_y=True)

    # Shuffle dataset
    n = X.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)

    X = X[index, :]
    y = y[index]

    # Split into train and test folds
    train = 0.75
    k = round(n * train)

    Xtrain, ytrain = X[:k, :], y[:k]
    Xtest, ytest = X[k:, :], y[k:]

    # List of hyperparameters
    alpha = [1e-3, 1e-2, 1e-1]
    k = [10 ** i for i in range(len(str(k)))] + [k]

    model = sklearn.model_selection.GridSearchCV(
        closek.CloseKClassifier(loss="hinge"),
        {"alpha": alpha, "k": k},
        cv=5)
    model.fit(Xtrain, ytrain)
    print("Close-k (hinge):     " + str(model.score(Xtest, ytest)))

    model = sklearn.model_selection.GridSearchCV(
        closek.CloseKClassifier(loss="log"),
        {"alpha": alpha, "k": k},
        cv=5)
    model.fit(Xtrain, ytrain)
    print("Close-k (logistic):  " + str(model.score(Xtest, ytest)))

    model = sklearn.model_selection.GridSearchCV(
        sklearn.linear_model.SGDClassifier(loss="hinge", max_iter=1000),
        {"alpha": alpha},
        cv=5)
    model.fit(Xtrain, ytrain)
    print("Linear SVM:          " + str(model.score(Xtest, ytest)))

    model = sklearn.model_selection.GridSearchCV(
        sklearn.linear_model.SGDClassifier(loss="log", max_iter=1000),
        {"alpha": alpha},
        cv=5)
    model.fit(Xtrain, ytrain)
    print("Logistic Regression: " + str(model.score(Xtest, ytest)))
