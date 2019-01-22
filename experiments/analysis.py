#!/usr/bin/env python

"""Creates figures and tables in paper."""

import math
import os
import pickle
import random
import statistics

import numpy as np
import torch
import scipy.stats

import util


def main():
    """Creates figures and tables in paper."""
    summarize_datasets("fig/summary")

    comparison("log/linear1", figroot="fig/linear1_combined_", dataset="combined")
    comparison("log/dense1", figroot="fig/dense1_combined_", dataset="combined")
    comparison("log_atk/linear1", figroot="fig/linear1_atk_", dataset="atk")
    comparison("log/linear1", figroot="fig/linear1_pmlb_binary_", dataset="pmlb_binary")
    comparison("log/dense1", figroot="fig/dense1_pmlb_binary_", dataset="pmlb_binary")
    comparison("log/linear1", figroot="fig/linear1_openml_binary_", dataset="openml_binary")
    comparison("log/dense1", figroot="fig/dense1_openml_binary_", dataset="openml_binary")

    simulation("simulation/spambase/outliers/", "Outliers", [0, 1, 2, 8, 16, 32, 64, 128, 256], "fig/outlier_spambase", 0.55, 0.95)
    simulation("simulation/spambase/dup0/", "Duplicates", [0, 800, 1600, 3200, 6400, 12800], "fig/dup0_spambase", 0.85, 1.0)
    simulation("simulation/spambase/dup1/", "Duplicates", [0, 800, 1600, 3200, 6400, 12800], "fig/dup1_spambase")
    simulation("simulation/spambase/hard0/", "Ambiguous Examples", [0, 100, 200, 400, 800], "fig/hard0_spambase", 0.7, 0.95)
    simulation("simulation/spambase/hard1/", "Ambiguous Examples", [0, 100, 200, 400, 800], "fig/hard1_spambase")


def summarize_datasets(root=None):
    """
    Create plot showing size and dimensions of datasets.

    This creates Figure 2.

    Args:
        root (str): where to save figure
    """
    import matplotlib.pyplot as plt

    jitter = 0.1  # small amount of jitter added to avoid overlap
    scale = 0.25

    random.seed(0)

    ds = util.pmlb_binary_classification_dataset_names()
    n1 = np.zeros(len(ds))
    d1 = np.zeros(len(ds))
    for (i, dataset) in enumerate(ds):
        X, y = util.load(dataset)
        n1[i], d1[i] = X.shape
        print(dataset)
        print(n1[i], d1[i])

        n1[i] *= math.exp(random.uniform(-jitter, jitter))
        d1[i] *= math.exp(random.uniform(-jitter, jitter))

    ds = util.openml_binary_classification_dataset_names()
    n2 = np.zeros(len(ds))
    d2 = np.zeros(len(ds))
    for (i, dataset) in enumerate(ds):
        X, y = util.load(dataset)
        n2[i], d2[i] = X.shape
        print(dataset)
        print(n2[i], d2[i])

        n2[i] *= math.exp(random.uniform(-jitter, jitter))
        d2[i] *= math.exp(random.uniform(-jitter, jitter))

    ds = util.atk_dataset_names()
    n3 = np.zeros(len(ds))
    d3 = np.zeros(len(ds))
    for (i, dataset) in enumerate(ds):
        X, y = util.load(dataset)
        n3[i], d3[i] = X.shape
        print(dataset)
        print(n3[i], d3[i])

        n3[i] *= math.exp(random.uniform(-jitter, jitter))
        d3[i] *= math.exp(random.uniform(-jitter, jitter))

    xmin = 1
    xmax = max(max(n1), max(n2), max(n3)) * 2
    ymin = 1
    ymax = max(max(d1), max(d2), max(d3)) * 2

    util.latexify()

    fig = plt.figure(figsize=(math.log(xmax / xmin) * scale, math.log(ymax / ymin) * scale))
    ax = plt.gca()

    ax.scatter(n1, d1, s=3)
    ax.scatter(n2, d2, s=3)
    ax.scatter(n3, d3, s=3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.title("Summary of Datasets")
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("\\# Data Points")
    plt.ylabel("Dimensions")
    plt.legend(["PMLB", "OpenML", "Fan et al."], loc="best")
    plt.tight_layout()

    if root is not None:
        plt.savefig(root + ".png")
        plt.savefig(root + ".pdf")
    else:
        plt.show()


def comparison(root="log/linear1", epsilon=0.02, figroot=None, degree=1, dataset="pmlb_binary"):
    """
    Generates figures and tables comparing the methods.

    This creates Figures 3 and 4, along with all tables (the formatting in the tables require
    some post-processing).

    Args:
        root (str): location of logs
        epsilon (float): threshold for difference in accuracy
        figroot (str): where to save figure
        degree (int): degree of polynomial used for experiments
        dataset (str): name of datasets used
    """
    import matplotlib.pyplot as plt

    total = 0

    methods = ["logistic_close", "logistic_close_decay", "logistic_atk", "logistic_average", "logistic_top", "hinge_close", "hinge_close_decay", "hinge_atk", "hinge_average", "hinge_top"]

    if dataset == "atk":
        dataset_names = eval("util." + dataset + "_dataset_names()")
    elif dataset == "combined":
        dataset_names = util.pmlb_binary_classification_dataset_names() + util.openml_binary_classification_dataset_names()
    else:
        dataset_names = eval("util." + dataset + "_classification_dataset_names()")

    with open(figroot + "acc.tex", "w") as file:  # create table of individual accuracies for each dataset/method (Table 3)
        header(file, methods)

        c = np.zeros((len(methods), len(methods)), np.int64)
        cp = np.zeros((len(methods), len(methods)), np.int64)
        near_best = np.zeros((len(methods)), np.int64)
        near_best_own = np.zeros((len(methods)), np.int64)
        result = {}
        full = {}

        for dataset in dataset_names:
            complete = True

            print(dataset)
            res = {}
            f = {}

            for method in methods:
                print(method)
                res[method] = load_results(root + "/" + dataset + "/" + method)
                if res[method] == []:
                    complete = False
                    continue
                assert(len(res[method]) == 1)
                res[method] = res[method][0]
                f[method] = [x for x in res[method]["trial"] if "best" in x]

                res[method] = [x["best"] for x in res[method]["trial"] if "best" in x]
                print(res[method])
                if res[method] != []:
                    print(statistics.mean(res[method]))
                    if len(res[method]) > 1:
                        print(statistics.stdev(res[method]))
                if res[method] == []:
                    complete = False


            if complete:
                total += 1
                best = max([statistics.mean(r) for r in res.values()])
                best_by_method = {ind_loss(m): -1 for m in methods}
                for il in best_by_method.keys():
                    best_by_method[il] = max([statistics.mean(res[m]) for m in res if il == ind_loss(m)])
                for (i, mi) in enumerate(methods):
                    if statistics.mean(res[mi]) + epsilon >= best:
                        near_best[i] += 1
                    if statistics.mean(res[mi]) + epsilon >= best_by_method[ind_loss(mi)]:
                        near_best_own[i] += 1
                file.write(dataset)
                for (i, mi) in enumerate(methods):
                    file.write(" & {:.2f}".format(100 - 100 * statistics.mean(res[mi])))
                file.write(" \\\\\n")
                for (i, mi) in enumerate(methods):
                    print(mi)
                    for (j, mj) in enumerate(methods):
                        if statistics.mean(res[mi]) - epsilon >= statistics.mean(res[mj]):
                            c[i][j] += 1
                        ttest = scipy.stats.ttest_ind(res[mi], res[mj])
                        if ttest.statistic > 0 and ttest.pvalue < 0.05:
                            cp[i][j] += 1

                if len(res["logistic_average"]) >= 1 and len(res["logistic_close_decay"]) >= 1:
                    result[dataset] = res
                    full[dataset] = f
                if len(result) == 1:
                    pass
                    # break

            print(c / total)
            print(cp / total)

            print()
        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")
        file.write("\\end{center}\n")

    print(total)
    cepsilon = c / total
    cp = cp / total
    for (c, filename) in [(cepsilon, "comparision.tex"), (cp, "comparisonp.tex")]:
        # Create table with pairwise improvement (either by p-value or raw increase in accuracy)
        # (Tables 1, 2, 4, and 5)
        with open(figroot + filename, "w") as f:
            header(f, methods)

            m = max([len(m) for m in methods])
            prev = None
            for (i, mi) in enumerate(methods):
                ind = ind_loss(mi)
                if prev is not None and ind != prev:
                    f.write("\\midrule\n")
                if prev is None or ind != prev:
                    f.write("\\multirow{?}{*}{Model}\n")
                prev = ind

                f.write(("& {:<" + str(m) + "}").format(mi.replace("_", " ")))
                for (j, mj) in enumerate(methods):
                    if i == j:
                        f.write(" &     ")
                    else:
                        f.write(" & {:.2f}".format(c[j][i]))
                f.write(" \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{center}\n")

    with open(figroot + "near.tex", "w") as f:
        near_best = near_best / total
        near_best_own = near_best_own / total
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{2}{c}{Compared against} \\\\\n")
        f.write("\\cmidrule(lr){2-3}\n")
        f.write("Method & All  & Own \\\\\n")
        f.write("\\midrule\n")
        for (i, mi) in enumerate(methods):
            f.write(("{:<" + str(m) + "} & {:.2f} & {:.2f}\\\\\n").format(mi.replace("_", " "), near_best[i], near_best_own[i]))
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}")
        f.write("\\end{center}\n")

    util.latexify()

    def extract(trials):
        """
        Returns the best results (by validation score) from a list of trials.
        
        Each trial runs a sweep over hyperparemters.
        """
        temp = []
        for trial in trials:
            best = None
            for hp in trial:
                if hp == "best":
                    continue
                res = trial[hp]
                if best is None or ((res["valid"] >= best["valid"]) if ("valid" in res) else (res["train"] >= best["train"])):
                    best = res
            temp.append(best)
        return temp

    def summarize(logfile, trials, X, y):
        """
        Extracts statistics about the performance of a method.

        Most of these statistics were not actually used in the paper (only for preliminary
        analysis).
        """
        potential = []
        minority = []
        majority = []
        hyperparams = []
        for (i, trial) in enumerate(trials):
            best = None
            for hp in trial:
                if hp == "best":
                    continue
                res = trial[hp]
                if best is None or ((res["valid"] >= best["valid"]) if ("valid" in res) else (res["train"] >= best["train"])):
                    best = res
                    best_hp = hp
            # print(best)
            model = load_model(logfile, i, best_hp)
            hyperparams.append(best_hp)

            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)

            Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = util.split(X, y)

            with torch.no_grad():
                pred = model(torch.Tensor(Xtest)).numpy().squeeze()

            perm = np.argsort(pred)
            pred = pred[perm]
            ytest = ytest[perm]

            n = ytest.shape[0]
            correct = ((pred > 0) == ytest).astype(np.float)
            test = statistics.mean(correct)
            possible = (sum(ytest) + np.concatenate((np.array([0]), np.cumsum(1 - 2 * ytest)))) / n
            ind = np.argmax(possible)

            thresh = np.concatenate((np.array([-np.inf]), pred, np.array([np.inf])))
            cutoff = (thresh[ind] + thresh[ind + 1]) / 2

            minority.append(np.sum(np.logical_and(pred > 0, ytest)) / np.sum(ytest))
            majority.append(np.sum(np.logical_and(pred <= 0, 1 - ytest)) / np.sum(1 - ytest))

            with torch.no_grad():
                pred = model(torch.Tensor(Xtest)).numpy().squeeze()
            correct = ((pred > 0) == ytest).astype(np.float)
            test_acc = statistics.mean(correct)

            correct = ((pred > cutoff) == ytest).astype(np.float)
            best = statistics.mean(correct)
            n = ytest.shape[0]

            best = max((sum(ytest) + np.concatenate((np.array([0]), np.cumsum(1 - 2 * ytest)))) / n)

            potential.append((best - test_acc) / n)
        return potential, minority, majority, hyperparams

    test_average  = np.empty(len(full))
    test_average_hinge  = np.empty(len(full))
    test_atk      = np.empty(len(full))
    test_close    = np.empty(len(full))
    test_close_hinge    = np.empty(len(full))
    test_average_std  = np.empty(len(full))
    test_atk_std      = np.empty(len(full))
    test_close_std    = np.empty(len(full))
    train_average = np.empty(len(full))
    train_atk     = np.empty(len(full))
    train_close   = np.empty(len(full))
    train_average_std = np.empty(len(full))
    train_atk_std     = np.empty(len(full))
    train_close_std   = np.empty(len(full))
    examples = np.empty(len(full))
    dimensions = np.empty(len(full))
    imbalance = np.empty(len(full))
    potential     = np.empty(len(full))

    test_average_minority  = np.empty(len(full))
    test_atk_minority      = np.empty(len(full))
    test_close_minority    = np.empty(len(full))
    test_average_majority  = np.empty(len(full))
    test_atk_majority      = np.empty(len(full))
    test_close_majority    = np.empty(len(full))

    selected_k    = np.empty(len(full))

    improvement_on = {}
    for (i, ds) in enumerate(full):
        # Accumulate information about each method over all datasets
        print(ds)
        X, y = util.load(ds)
        n, d = X.shape

        X = X - np.mean(X, 0)
        X = X / np.maximum(np.std(X, 0), 1e-5)

        X = util.PolynomialFeature(degree)(torch.Tensor(X)).numpy()
        examples[i], dimensions[i] = X.shape
        larger = max(sum(y == 0), sum(y == 1))
        smaller = examples[i] - larger
        imbalance[i] = larger / smaller

        test_average[i] = statistics.mean(map(lambda t: t["best"], full[ds]["logistic_average"]))
        test_average_hinge[i] = statistics.mean(map(lambda t: t["best"], full[ds]["hinge_average"]))
        test_atk[i]     = statistics.mean(map(lambda t: t["best"], full[ds]["logistic_atk"]))
        test_close[i]   = statistics.mean(map(lambda t: t["best"], full[ds]["logistic_close_decay"]))
        test_close_hinge[i] = statistics.mean(map(lambda t: t["best"], full[ds]["hinge_close_decay"]))
        try:
            test_average_std[i] = statistics.stdev(map(lambda t: t["best"], full[ds]["logistic_average"]))
            test_atk_std[i]     = statistics.stdev(map(lambda t: t["best"], full[ds]["logistic_atk"]))
            test_close_std[i]   = statistics.stdev(map(lambda t: t["best"], full[ds]["logistic_close_decay"]))
        except statistics.StatisticsError:
            pass

        improvement_on[ds] = test_close[i] - test_average[i]

        train_average[i] = statistics.mean(map(lambda x: x["train"], extract(full[ds]["logistic_average"])))
        train_atk[i]     = statistics.mean(map(lambda x: x["train"], extract(full[ds]["logistic_atk"])))
        train_close[i]   = statistics.mean(map(lambda x: x["train"], extract(full[ds]["logistic_close_decay"])))
        try:
            train_average_std[i] = statistics.stdev(map(lambda x: x["train"], extract(full[ds]["logistic_average"])))
            train_atk_std[i]     = statistics.stdev(map(lambda x: x["train"], extract(full[ds]["logistic_atk"])))
            train_close_std[i]   = statistics.stdev(map(lambda x: x["train"], extract(full[ds]["logistic_close_decay"])))
        except statistics.StatisticsError:
            pass
        
        average_summary = summarize(root + "/" + ds + "/logistic_average", full[ds]["logistic_average"], X, y)
        atk_summary = summarize(root + "/" + ds + "/logistic_atk", full[ds]["logistic_atk"], X, y)
        close_summary = summarize(root + "/" + ds + "/logistic_close", full[ds]["logistic_close"], X, y)

        potential[i] = statistics.mean(average_summary[0])

        test_average_minority[i]  = statistics.mean(average_summary[1])
        test_atk_minority[i]      = statistics.mean(atk_summary[1])
        test_close_minority[i]    = statistics.mean(close_summary[1])

        test_average_majority[i]  = statistics.mean(average_summary[2])
        test_atk_majority[i]      = statistics.mean(atk_summary[2])
        test_close_majority[i]    = statistics.mean(close_summary[2])

        selected_k[i] = statistics.mean(list(map(lambda x: x[1], close_summary[3]))) / n

    improvement_on = [(ds, improvement_on[ds]) for ds in sorted(improvement_on, key=improvement_on.get, reverse=True)]

    # Figure 3a
    plt.figure(figsize=(1.5, 1.5))
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.scatter(test_average, test_close, s=1)
    plt.title("")
    plt.axis([0.5, 1, 0.5, 1])
    plt.xlabel("Accuracy of Average")
    plt.ylabel("Accuracy of Algorithm 1")
    plt.tight_layout()
    if figroot is None:
        plt.show()
    else:
        plt.savefig(figroot + "logistic_improvement.pdf")

    # Figure 3b
    plt.figure(figsize=(1.5, 1.5))
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.scatter(test_average_hinge, test_close_hinge, s=1)
    plt.title("")
    plt.axis([0.5, 1, 0.5, 1])
    plt.xlabel("Accuracy of Average")
    plt.ylabel("Accuracy of Algorithm 1")
    plt.tight_layout()
    if figroot is None:
        plt.show()
    else:
        plt.savefig(figroot + "hinge_improvement.pdf")


    # Figure 4a
    plt.figure(figsize=(1.5, 1.5))
    plt.hist(selected_k, bins=20)
    plt.title("")
    plt.xlabel("$k^*/n$")
    plt.ylabel("Fraction of Datasets")
    plt.tight_layout()
    if figroot is None:
        plt.show()
    else:
        plt.savefig(figroot + "selected_k.pdf")

    # Figure 4b
    plt.figure(figsize=(1.5, 1.5))
    plt.scatter(selected_k, test_close - test_average, s=1)
    plt.title("")
    plt.xlabel("$k^*/n$")
    plt.ylabel("Improvement")
    plt.tight_layout()
    if figroot is None:
        plt.show()
    else:
        plt.savefig(figroot + "improvement_vs_k.pdf")


def simulation(root, xlabel, params, figroot=None, ymin=0.5, ymax=1):
    """
    Creates figures for the simulations (Figure 5).


    Args:
        root (str): location of logs
        params (list of int): parameters to plot
        figroot (str): where to save figure
        ymin (flat): lower limit for y-axis
        ymax (flat): upper limit for y-axis
    """
    import matplotlib.pyplot as plt

    method = ["close", "atk", "average", "top"]

    util.latexify()
    plt.figure(figsize=(1.8, 1.8))
    ax = plt.gca()

    h = []
    for m in method:
        a = []
        print(m)
        for o in params:
            res = load_results(root + "/" + m + "_" + str(o))
            a.append(statistics.mean([x["best"] for x in res[0]["trial"] if "best" in x]))
            print(a)
        if m == "close_decay":
            m = "decay"
        l, = ax.plot(params, a, linewidth=1, label=m)
        h.append(l)

    temp = root.split("/")
    dataset = temp[1]
    mode = temp[2]
    X, y = util.load(dataset)
    n0 = sum(y == 0)
    n1 = sum(y == 1)
    if mode == "outliers":
        n0 = [n0 for _ in params]
        n1 = [n1 for _ in params]
    elif mode == "dup0":
        n0 = [n0 + p for p in params]
        n1 = [n1 for p in params]
    elif mode == "dup1":
        n0 = [n0 for p in params]
        n1 = [n1 + p for p in params]
    elif mode == "hard0":
        n0 = [n0 for p in params]
        n1 = [n1 + p for p in params]
    elif mode == "hard1":
        n0 = [n0 + p for p in params]
        n1 = [n1 for p in params]
    else:
        raise ValueError()
    a = [max(n0[i], n1[i]) / (n0[i] + n1[i]) for i in range(len(params))]
    l, = ax.plot(params, a, linewidth=1, color="black", linestyle="--", label="majority")
    h.append(l)

    plt.title("")
    plt.axis([0, max(params), ymin, ymax])
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    if figroot is None:
        plt.show()
    else:
        plt.savefig(figroot + ".pdf")

    ax.legend(loc="center left")
    plt.axis("off")
    plt.axis([-1, -1, -1, -1])
    if figroot is not None:
        plt.savefig(os.path.dirname(figroot) + "/legend.pdf")


def ind_loss(method):
    """
    Extract individual loss from full name of method.

    Args:
        method (str): full name of method

    Returns:
        str: individual loss
    """
    if method[:6] == "hinge_":
        return "Hinge"
    if method[:9] == "logistic_":
        return "Logistic"
    if method[:7] == "sklearn":
        return "sklearn"


def agg_loss(method):
    """
    Extract aggregate loss from full name of method.

    Args:
        method (str): full name of method

    Returns:
        str: aggregate loss
    """
    if method[:6] == "hinge_":
        return method[6:]
    if method[:9] == "logistic_":
        return method[9:]
    if method[:7] == "sklearn":
        return method[7:]


def header(f, methods):
    """
    Write header for LaTex tables in paper.

    Some manual manipulation was done for final version.

    Args:
        f (file): file to write header to
        methods (list of str): list of full method names

    Returns:
        str: aggregate loss
    """
    f.write("\\begin{center}\n")
    f.write("\\begin{tabular}{ll" + len(methods) * "r" + "}\n")
    f.write("\\toprule\n")
    f.write(" & ")
    prev = None
    count = [1]
    for (j, mj) in enumerate(methods + [None]):
        if mj is None:
            pass
        else:
            mj = ind_loss(mj)

        if mj != prev:
            if prev is not None:
                f.write(" & \\multicolumn{" + str(count[-1]) + "}{c}{" + prev + "}")
            prev = mj
            if mj is not None:
                count.append(1)
        else:
            count[-1] += 1
        prev = mj
    f.write(" \\\\\n")

    print(count)
    start = 1
    for i in range(1, len(count)):
        start += count[i - 1]
        f.write("\\cmidrule(lr){" + str(start + 1) + "-" + str(start + count[i]) + "}")
    f.write(" \n")
    for (j, mj) in enumerate(methods):
        mj = agg_loss(mj)
        f.write(" & " + mj.replace("_", " "))
    f.write(" \\\\\n")
    f.write("\\midrule\n")


def load_results(filename):
    """
    Returns dictionary of parsed results from log file.

    Args:
        filename (string): name of log file

    Returns:
        dict: parsed results
    """
    picklename = filename + ".pkl"

    if not os.path.isfile(filename) and not os.path.isfile(picklename):
        return []
    if os.path.isfile(picklename) and (not os.path.isfile(filename) or os.path.getctime(filename) < os.path.getctime(picklename)):
        with open(picklename, "rb") as f:
            result = pickle.load(f)
    else:
        with open(filename) as f:
            result = []
            f = [l[26:] for l in f if l[:4] == "INFO" or "Namespace" in l]
            for l in f:
                if "Namespace" in l:
                    pass
                elif l[:6] == "Mode: ":
                    if not result:
                        result.append({})
                        result[-1]["mode"] = l[6:-1]
                        result[-1]["trial"] = []
                elif l[:7] == "Trial #":
                    trial = int(l[7:-2])
                    if len(result[-1]["trial"]) + 1 == trial:
                        result[-1]["trial"].append({})
                    elif len(result[-1]["trial"]) == trial:
                        result[-1]["trial"][trial - 1] = {}
                    else:
                        break
                elif l[:14] == "    Accuracy: ":
                    result[-1]["trial"][trial - 1]["best"] = float(l[14:])
                elif l[:22] == "    Average Accuracy: ":
                    pass
                elif l[:19] == "    Average Error: ":
                    pass
                elif l[:28] == "Early termination at epoch #":
                    pass
                else:
                    i = l.find(":")
                    result[-1]["trial"][trial - 1][eval(l[:i])] = eval(l[i + 1:])
        with open(picklename, "wb") as f:
            pickle.dump(result, f)
    return result


def load_model(logfile, trial, hp):
    """
    Returns the torch.nn.Module from a specific trial and hyperparameter choice.
    """
    with open(logfile + "_checkpoints/" + "_".join([str(trial)] + list(map(str, hp))) + ".pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
