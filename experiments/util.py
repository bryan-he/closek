"""Simple classes and functions."""

import os
import random
import pickle
import pathlib
import argparse
import datetime
import distutils.util
import logging
import logging.config

import numpy as np
import torch
import matplotlib
import pmlb
import openml


def loglevel(level):
    """
    Converts a string representing the logging level to corresponding number.

    This is primarily used as the type of an argument in argparse. See
    https://docs.python.org/3/library/logging.html#logging-levels for a list
    of logging levels.

    Args:
        level (str): logging level

    Returns:
        int: Numeric logging level

    Raises:
        ValueError: If `level` is not a recognized logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level


class MultilineFormatter(logging.Formatter):
    """Formatter for `logging` package to repeat header for multiline records.

    This class is taken from
    https://mail.python.org/pipermail/python-list/2010-November/591474.html.
    """
    def format(self, record):
        """Converts a record to properly formatted string."""
        string = logging.Formatter.format(self, record)
        header, _ = string.split(record.message)
        string = string.replace('\n', '\n' + ' '*len(header))
        return string


def pmlb_binary_classification_dataset_names():
    """Returns list of binary classification datasets in PMLB."""
    try:
        name = pickle.load(open(".pmlb/bcdn.pkl", "rb"))
    except FileNotFoundError:
        pathlib.Path(".pmlb").mkdir(parents=True, exist_ok=True)

        name = []
        for dataset in pmlb.classification_dataset_names:
            X, y = pmlb.fetch_data(dataset, return_X_y=True, local_cache_dir=".pmlb")
            if np.unique(y).size == 2:
                name.append(dataset)
        pickle.dump(name, open(".pmlb/bcdn.pkl", "wb"))
    return name


def pmlb_multiclass_classification_dataset_names():
    """Returns list of multiclass classification datasets in PMLB."""
    try:
        name = pickle.load(open(".pmlb/mcdn.pkl", "rb"))
    except FileNotFoundError:
        pathlib.Path(".pmlb").mkdir(parents=True, exist_ok=True)

        name = []
        for dataset in pmlb.classification_dataset_names:
            X, y = pmlb.fetch_data(dataset, return_X_y=True, local_cache_dir=".pmlb")
            if np.unique(y).size != 2:
                name.append(dataset)
        pickle.dump(name, open(".pmlb/mcdn.pkl", "wb"))
    return name


def openml_binary_classification_dataset_names():
    """Returns list of binary classification datasets in OpenML100."""
    try:
        name = pickle.load(open(".openml/bcdn.pkl", "rb"))
    except FileNotFoundError:
        pathlib.Path(".openml").mkdir(parents=True, exist_ok=True)

        name = []
        benchmark_suite = openml.study.get_study('OpenML100', 'tasks')
        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y()
            if np.unique(y).size == 2:
                name.append(str(task_id))
        pickle.dump(name, open(".openml/bcdn.pkl", "wb"))
    return name


def openml_multiclass_classification_dataset_names():
    """Returns list of multiclass classification datasets in OpenML100."""
    try:
        name = pickle.load(open(".openml/mcdn.pkl", "rb"))
    except FileNotFoundError:
        pathlib.Path(".openml").mkdir(parents=True, exist_ok=True)

        name = []
        benchmark_suite = openml.study.get_study('OpenML100', 'tasks')
        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y()
            if np.unique(y).size != 2:
                name.append(str(task_id))
        pickle.dump(name, open(".openml/mcdn.pkl", "wb"))
    return name


def atk_dataset_names():
    """Returns list of datasets used by
    Fan, Yanbo, et al. "Learning with average top-k loss."
    Advances in Neural Information Processing Systems. 2017.
    """
    return [
        "monk3",
        "phoneme",
        "madelon",
        "spambase",
        "titanic",
        "australian",
        "splice3175",
        "german",
        "german-num",
        ]


def latexify():
    """
    Sets `matplotlib` parameters to create figures that match LaTeX fonts.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html.
    """
    params = {'backend': 'pdf',
              'axes.labelsize':  8,
              'font.size':       8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
             }
    matplotlib.rcParams.update(params)


def get_parser():
    """Returns the parser for running experiments from the command line."""
    parser = argparse.ArgumentParser(description="close-k experiments")

    parser.add_argument("dataset", type=str,
                        help="dataset to use")

    parser.add_argument("--trials", "-t", type=int, default=1,
                        help="number of folds")
    parser.add_argument("--seed", "-s", type=int, default=0,
                        help="seed for RNG")

    parser.add_argument("--degree", "-d", type=int, default=1,
                        help="degree of polynomial")
    parser.add_argument("--pca", type=int, default=None,
                        help="number of pca dimensions")

    parser.add_argument("--split", type=distutils.util.strtobool, default=True,
                        help="whether or not to validate")
    parser.add_argument("--render", type=distutils.util.strtobool, default=False, help="render")
    parser.add_argument("--synthetic", type=distutils.util.strtobool, default=False,
                        help="only use first dim")

    parser.add_argument("--outliers", type=int, default=0, help="number of outliers to insert")
    parser.add_argument("--dup0", type=int, default=0, help="number of times to duplicate 0 class")
    parser.add_argument("--dup1", type=int, default=0, help="number of times to duplicate 1 class")
    parser.add_argument("--hard0", type=int, default=0, help="number of times to flip 0 class")
    parser.add_argument("--hard1", type=int, default=0, help="number of times to flip 1 class")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--logistic", action="store_const", const="logistic", dest="loss",
                       default="logistic", help="use logistic loss")
    group.add_argument("--hinge", action="store_const", const="hinge", dest="loss",
                       help="use hinge loss")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--normal", action="store_const", const="normal", dest="multi",
                       default="normal", help="use multi-class normally")
    group.add_argument("--small", action="store_const", const="small", dest="multi",
                       help="train multi-class as one-against-smallest")
    group.add_argument("--large", action="store_const", const="large", dest="multi",
                       help="train multi-class as one-against-largest")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--linear", action="store_const", const="linear", dest="model",
                       default="linear", help="use linear decision")
    group.add_argument("--nn", action="store_const", const="nn", dest="model",
                       help="use neural net")
    group.add_argument("--dense", action="store_const", const="dense", dest="model",
                       help="use dense block")

    parser.add_argument("--mode", "-m", nargs="+",
                        choices=["average", "atk", "top", "maximal", "trunc", "border", "close",
                                 "close_decay", "drop", "sklearnlogistic", "sklearnsvm",
                                 "sklearnnb", "sklearnsgd"],
                        default=["average"], help="training mode")
    parser.add_argument("--labels", nargs="+", default=None, help="labels for figure")
    parser.add_argument("--threshold", default=None, type=float, help="threshold for close")

    parser.add_argument("--gpu", "-g", nargs="?", const=True, type=distutils.util.strtobool,
                        default=False, help="use GPU")
    parser.add_argument("--pretrain", "-p", type=distutils.util.strtobool, default=True,
                        help="initialize with linear regression")
    parser.add_argument("--normalize", "-n", type=distutils.util.strtobool, default=True,
                        help="normalize features")
    parser.add_argument("--save", nargs="?", const=True, type=distutils.util.strtobool,
                        default=False, help="save models")
    parser.add_argument("--restart", nargs="?", const=True, type=distutils.util.strtobool,
                        default=False, help="restart run")
    parser.add_argument("--batch", type=int, default=10000, help="batch size")
    parser.add_argument("--lr", type=float, default=1., help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--reg", "-r", type=float, nargs="+", default=None,
                        help="regularization parameters")
    parser.add_argument("--k", "-k", type=int, nargs="+", default=None,
                        help="number of data points")
    parser.add_argument("--tol", type=float, default=1e-3, help="tolerance for convergence")
    parser.add_argument("--log", "-l", type=loglevel, default=logging.DEBUG)
    parser.add_argument("--logfile", type=str,
                        default="log/" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        help="file to store logs")

    return parser


def configure_logger(level, logfile):
    """
    Basic configuration for logging format.

    Main settings:
      - Include time in header
      - Fix header for multiline messages (see `MultilineFormatter`)
      - Set logging level
      - Save log to file
    """
    cfg = dict(
        version=1,
        formatters={
            "f": {"()":
                      "util.MultilineFormatter",
                  "format":
                      "%(levelname)-8s [%(asctime)s] %(message)s",
                  "datefmt":
                      "%m/%d %H:%M:%S"}
            },
        handlers={
            "s": {"class": "logging.StreamHandler",
                  "formatter": "f",
                  "level": level},
            "f": {"class": "logging.FileHandler",
                  "formatter": "f",
                  "level": level,
                  "filename": logfile}
            },
        root={
            "handlers": ["s", "f"],
            "level": logging.NOTSET
            },
    )
    logging.config.dictConfig(cfg)


def load(dataset="monk3", multi="normal"):
    """
    Returns X (features) and y (classes) for a dataset.

    The dataset can be from PMLB, OpenML100, a .dat file in the data/ directory, or one of the
    synthetic examples from Figure 1.


    Args:
        dataset (str): name of the dataset
        multi (str): mode for processing multi-class problems
            There are three valid choices:
                - "normal": return multi-class problems normally
                - "small": convert multi-class problem into a smallest class against all problem
                - "large": convert multi-class problem into a largest class against all problem

    Returns:
        X (np.array): features the data points
        y (np.array): classes for the data points
    """

    try:
        # PMLB does not provide a simple way to check if a dataset is available
        # Just attempt to load, and continue through list of datasets if not found
        pathlib.Path(".pmlb").mkdir(parents=True, exist_ok=True)
        X, y = pmlb.fetch_data(dataset, return_X_y=True, local_cache_dir=".pmlb")
        isPMLB = True
    except ValueError:
        isPMLB = False

    if isPMLB:
        # PMLB data already loaded
        pass
    elif dataset in map(str, openml.study.get_study("OpenML100", "tasks").tasks):
        task = openml.tasks.get_task(dataset)
        X, y = task.get_X_and_y()
        X = X[:, sum(np.isnan(X)) == 0]
    elif os.path.isfile("data/" + dataset + ".dat"):
        # Datasets not in PMLB or OpenML (load from file)
        X = np.genfromtxt("data/" + dataset + ".dat", delimiter=",")
        X, y = X[:, :-1], X[:, -1].astype(np.int64)
    elif dataset == "easy":
        # Synthetic example for Figure 1a
        r1 = 225
        r2 =  25
        b  = 250
        X = np.concatenate([np.concatenate([0.10 * np.random.rand(r1, 1) - 1.0, 2 * np.random.rand(r1, 1) - 1], axis=1),
                            np.concatenate([0.25 * np.random.rand(r2, 1) - 0.2, 2 * np.random.rand(r2, 1) - 1], axis=1),
                            np.concatenate([0.10 * np.random.rand(b,  1) + 0.0, 2 * np.random.rand(b, 1) - 1], axis=1)])
        y = np.array((r1 + r2) * [1] + b * [0])
    elif dataset == "imbalance":
        # Synthetic example for Figure 1b
        r =  20
        b = 480
        X = np.concatenate([np.concatenate([1.05 * np.random.rand(r, 1) - 1.0, 2 * np.random.rand(r, 1) - 1], axis=1),
                            np.concatenate([1.00 * np.random.rand(b, 1) + 0.0, 2 * np.random.rand(b, 1) - 1], axis=1)])
        y = np.array(r * [1] + b * [0])
    elif dataset == "imbalance+outlier":
        # Synthetic example for Figure 1c
        r =  20
        b = 480
        X = np.concatenate([np.concatenate([1 * np.random.rand(r, 1) - 1.0, 2 * np.random.rand(r, 1) - 1], axis=1),
                            np.concatenate([1 * np.random.rand(b, 1) + 0.0, 2 * np.random.rand(b, 1) - 1], axis=1),
                            np.array([[-1.0, 0.0]])])
        y = np.array(r * [1] + b * [0] + [0])
    elif dataset == "overlap":
        # Synthetic example for Figure 1d
        r = 250
        b = 250
        X = np.concatenate([np.concatenate([1 * np.random.rand(r, 1) - 1, 2 * np.random.rand(r, 1) - 1], axis=1),
                            np.concatenate([2 * np.random.rand(b, 1) - 1, 2 * np.random.rand(b, 1) - 1], axis=1)])
        y = np.array(r * [1] + b * [0])
    else:
        raise ValueError("Dataset " + dataset + " is not recognized.")

    # Map classes down to 0 to (number_of_classes - 1)
    unique = np.unique(y)
    new = {old: new for (new, old) in enumerate(unique)}
    y = np.array([new[i] for i in y])

    if multi == "normal":
        # Treat multi-class problems normally
        pass
    else:
        # Convert multi-class problems to binary
        count = np.bincount(y)
        if multi == "small":
            # Smallest class against all
            count = -count
        elif multi == "large":
            # Largest class against all
            pass
        else:
            raise ValueError("Multi-class setting \"" + multi + "\" not recognized.")
        ind = np.argmax(count)
        y = (y == ind).astype(np.int)

    return X, y


def split(X, y, train=0.5, valid=0.25, test=0.25):
    """
    Split dataset into train, validation, and test folds.

    Args:
        X (np.array): features for the data points
        y (np.array): classes for the data points
        train (float): fraction of points in train fold
        valid (float): fraction of points in validation fold
        test (float): fraction of points in test fold

    Returns:
        Xtrain (np.array): train fold features
        ytrain (np.array): train fold classes
        Xvalid (np.array): validation fold features
        yvalid (np.array): validation fold classes
        Xtest (np.array): test fold features
        ytest (np.array): test fold classes

    Raises:
        ValueError: If `level` is not a recognized logging level.
    """
    if abs(train + valid + test - 1) > 1e-5:
        raise ValueError("train, valid, and test must add to approximately 1.")

    n = X.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)

    X = X[index, :]
    y = y[index]

    n1 = round(n * train)
    n2 = round(n * valid) + n1

    return X[:n1, :], y[:n1], X[n1:n2, :], y[n1:n2], X[n2:, :], y[n2:]


class PolynomialFeature(torch.nn.Module):
    """
    Neural network module for generating polynomial features.

    Attributes:
        degree (int): degree of pulynomial to generate
    """
    def __init__(self, degree):
        super(PolynomialFeature, self).__init__()
        self.degree = degree

    def forward(self, x):
        return torch.cat([x ** (i + 1) for i in range(self.degree)], dim=1)


class HingeLoss(torch.nn.Module):
    """Neural network module for computing hinge loss."""
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pred, target):
        return (1 - pred * target).clamp(0)


# For pre-training
def linear_regression(X, y, bias=True, rcond=None):
    """
    Computes a linear regression model.

    This can be used as the initial parameter when training classifiers with a
    linear decision boundary. This can help reduce the number of epochs needed.

    Args:
        X (np.array): features
        y (np.array): classes
        bias (bool, optional): whether or not to include bias term
        rcond (float, optional): cutoff for small singular values (for `np.linalg.pinv`)

    Returns:
        torch.nn.Linear: Model containing weights from linear regression

    Raises:
        ValueError: If class labels cannot be are not 0/1 or -1/1.
    """
    linear = torch.nn.Linear(X.shape[1], 1, bias)
    if bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    if all(np.unique(y) == np.array([0, 1])):
        y = 2 * y - 1
    if not all(np.unique(y) == np.array([-1, 1])):
        raise ValueError("Linear regression initialization encountered unknown class labels.")

    if rcond is None:
        w = np.dot(np.linalg.pinv(X), y)
    else:
        w = np.dot(np.linalg.pinv(X, rcond), y)

    linear.weight.data = torch.Tensor(w[:linear.weight.numel()].reshape(1, -1))
    if bias:
        linear.bias.data = torch.Tensor([w[-1]])

    return linear

def simulate_data(X, y, outliers=0, dup0=0, dup1=0, hard0=0, hard1=0):
    """
    Simulates variants of a dataset with outliers, class imbalance, or hard examples.

    These simulated variants are used for Figure 5 of the paper. The default
    arguments for the parameters results in the dataset being returned without
    modification.

    Args:
        X (np.array): features
        y (np.array): classes
        outliers (int): number of outliers to simulate
        dup0 (int): number of negative examples to duplicate
        dup1 (int): number of positive examples to duplicate
        hard0 (int): number of negative examples to convert into hard examples
        hard1 (int): number of positive examples to convert into hard examples

    Returns:
        np.array: features
        np.array: classes
    """
    # Options for simulated variants (Figure 5)

    if outliers != 0:
        # Add outliers, which are placed very far in the direction of one class,
        # but given the opposite label
        Xextra = np.empty((outliers, X.shape[1]), X.dtype)
        yextra = np.empty((outliers,), y.dtype)
        for i in range(outliers):
            while True:  # Sample two elements with different classes
                j = random.randint(0, X.shape[0] - 1)
                k = random.randint(0, X.shape[0] - 1)
                if y[j] != y[k]:
                    break
            # Place the outlier very far in the direction of example k,
            # but give it the same label as j
            alpha = 10  # Amount placed in direction of example k
            Xextra[i, :] = (1 - alpha) * X[j, :] + alpha * X[k, :]
            yextra[i] = y[j]
        X = np.concatenate([X, Xextra])
        y = np.concatenate([y, yextra])
    if dup0 != 0:
        # Add duplicated negative examples (same as one of the existing negatives) to simulate
        # class imbalance
        Xextra = np.empty((dup0, X.shape[1]), X.dtype)
        yextra = np.empty((dup0,), y.dtype)
        for i in range(dup0):
            while True:  # Sample a negative example
                j = random.randint(0, X.shape[0] - 1)
                if y[j] == 0:
                    break
            Xextra[i, :] = X[j, :]
            yextra[i] = y[j]
        X = np.concatenate([X, Xextra])
        y = np.concatenate([y, yextra])
    if dup1 != 0:
        # Add duplicated positive examples (same as one of the existing positive) to simulate
        # class imbalance
        Xextra = np.empty((dup1, X.shape[1]), X.dtype)
        yextra = np.empty((dup1,), y.dtype)
        for i in range(dup1):
            while True:  # Sample a positive example
                j = random.randint(0, X.shape[0] - 1)
                if y[j] == 1:
                    break
            Xextra[i, :] = X[j, :]
            yextra[i] = y[j]
        X = np.concatenate([X, Xextra])
        y = np.concatenate([y, yextra])
    if hard0 != 0:
        # Add copies of negative examples that are labeled positive, giving hard examples
        Xextra = np.empty((hard0, X.shape[1]), X.dtype)
        yextra = np.empty((hard0,), y.dtype)
        for i in range(hard0):
            while True:  # Sample a negative example
                j = random.randint(0, X.shape[0] - 1)
                if y[j] == 0:
                    break
            Xextra[i, :] = X[j, :]
            yextra[i] = 1
        X = np.concatenate([X, Xextra])
        y = np.concatenate([y, yextra])
    if hard1 != 0:
        # Add copies of positive examples that are labeled negative, giving hard examples
        Xextra = np.empty((hard1, X.shape[1]), X.dtype)
        yextra = np.empty((hard1,), y.dtype)
        for i in range(hard1):
            while True:  # Sample a positive example
                j = random.randint(0, X.shape[0] - 1)
                if y[j] == 1:
                    break
            Xextra[i, :] = X[j, :]
            yextra[i] = 0
        X = np.concatenate([X, Xextra])
        y = np.concatenate([y, yextra])

    return X, y


def render(X, y, model=None, root=None, X_remaining=None, transpose=True, labels=None):
    """
    Creates figure with scatterplot of data points and decision boundaries of models.

    This function is used to create Figure 1.

    Args:
        X (np.array): features
        y (np.array): classes
        root (str, optional): root for filename (defaults to showing on screen)
        model (list of tuples (str, torch.nn.module), optional): list of models to be plotted
        X_remaining (np.array, optional): additional dimensions to plot
        transpose (bool): transpose the axes in the plot
        labels (list of str): list of string for labeling contours (defaults to names in `model`)
    """
    # Conditional import so that rest of code can be run on systems without libraries for plotting
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Shuffle points (in case points overlap)
    n = X.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)
    X = X[index, :]
    y = y[index]
    if X_remaining is not None:
        X_remaining = X_remaining[index, :]

    latexify()

    plt.figure(figsize=(1.65, 1.65))

    colormap = sns.color_palette("pastel")
    colormap = [colormap[0], colormap[2]]
    color = [colormap[i] for i in y]

    if X_remaining is not None:
        X = np.concatenate([X, X_remaining], axis=1)

    xmin = -1.1
    xmax = +1.1
    ymin = -1.1
    ymax = +1.1

    if transpose:
        xmin, ymin = ymin, xmin
        xmax, ymax = ymax, xmax

    if transpose:
        plt.scatter(X[:, 1], X[:, 0], c=color, s=0.5)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=color, s=0.5)

    if model is not None:
        rows = 500
        cols = 500
        row = np.linspace(ymin, ymax, rows)
        col = np.linspace(xmin, xmax, cols)

        if X_remaining is None:
            if transpose:
                Xall = np.concatenate([np.repeat(row, cols).reshape(-1, 1),
                                       np.tile(col, rows).reshape(-1, 1)], 1)
            else:
                Xall = np.concatenate([np.tile(col, rows).reshape(-1, 1),
                                       np.repeat(row, cols).reshape(-1, 1)], 1)
        else:
            if transpose:
                Xall = np.repeat(row, cols).reshape(-1, 1)
            else:
                Xall = np.tile(col, rows).reshape(-1, 1)
        Xall = torch.Tensor(Xall)
        if not isinstance(model, list):
            model = [model]

        for (i, (name, m)) in enumerate(model):
            d = m(Xall).cpu().detach().numpy().reshape(rows, cols)

            CS = plt.contour(col, row, d, [0], colors=["black"], linewidths=1)

            fmt = {}
            if labels is None:
                strs = [name]
            else:
                strs = [labels[i]]
            for l, s in zip(CS.levels, strs):
                fmt[l] = s

            plt.clabel(CS, inline=True, fmt=fmt, fontsize=10, manual=[(0, 0)])

    plt.title("")
    plt.axis([xmin, xmax, ymin, ymax])
    plt.yticks([-1, 0, 1])
    plt.tight_layout()

    if root is not None:
        plt.savefig(root + ".pdf")
    else:
        plt.show()
