#!/usr/bin/env python

"""This module is used to run the experiments in the paper."""

import random
import math
import os
import pathlib
import itertools
import pickle
import copy
import traceback
import logging

import numpy as np
import torch
import sklearn.decomposition
import sklearn.linear_model
import sklearn.naive_bayes
import file_read_backwards

import util
import dense


def main(args=None):
    # Parse command line arguments
    parser = util.get_parser()
    args = parser.parse_args(args)

    # Automatically create directory for logfile (if needed)
    pathlib.Path(os.path.dirname(args.logfile)).mkdir(parents=True, exist_ok=True)

    # Basic configuration for logging
    util.configure_logger(args.log, args.logfile)

    logger = logging.getLogger(__name__)
    try:
        logger.info(args)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.loss == "logistic":
            loss = torch.nn.SoftMarginLoss(reduction="none")
        elif args.loss == "hinge":
            loss = util.HingeLoss()
        else:
            raise ValueError("Loss " + args.loss + " is not recognized")

        # Automatically set threshold if it is needed
        if args.threshold is None and set(["close", "close_decay", "drop"]).intersection(args.mode):
            if args.loss == "logistic":
                args.threshold = math.log(2)
            elif args.loss == "hinge":
                args.threshold = 1
            else:
                raise ValueError("Loss " + mode + " is not recognized")

        if args.gpu and not torch.cuda.is_available():
            logger.warning("GPU is not available.")
            args.gpu = False

        X, y = util.load(args.dataset, multi=args.multi)

        if args.normalize:  # Set input to be zero-mean and unit variance
            X = X - np.mean(X, 0)
            # Avoid division-by-zero in special case where one feature is constant
            X = X / np.maximum(np.std(X, 0), 1e-5)

        X, y = util.simulate_data(X, y, args.outliers, args.dup0, args.dup1, args.hard0, args.hard1)

        if args.pca is not None:
            # Project onto top principal components
            pca = sklearn.decomposition.PCA(args.pca)
            pca.fit(X)
            X = pca.transform(X)

        # Use a polynomial feature transform
        X = util.PolynomialFeature(args.degree)(torch.Tensor(X)).numpy()

        if args.synthetic:
            # Only consider one dimension for synthetic plots
            X, X_remaining = X[:, 0:1], X[:, 1:]
        else:
            X_remaining = None

        if args.restart:
            # This checks for an existing log file and determines which was the last completed
            # trial. The seed and number of trials are set to continue from this point.
            with file_read_backwards.FileReadBackwards(args.logfile) as f:
                try:
                    # Search for the last line stating a trial number in the log file
                    l = next(l for l in f if "Trial #" in l)
                    trial = int(l[33:-1])  # trial that was killed
                    newseed = args.seed + trial - 1
                    newtrials = args.trials - trial + 1
                    logger.warning("Detected previous run that failed on trial #%d\n"
                                   "Changing seed from %d to %d\n"
                                   "Changing trials from %d to %d",
                                   trial, args.seed, newseed, args.trials, newtrials)
                    args.seed = newseed
                    args.trials = newtrials
                except StopIteration:
                    # No failed run detected
                    pass

        solution = [None for _ in args.mode]
        for (m, mode) in enumerate(args.mode):
            logger.info("Mode: " + mode)
            accuracy = np.empty(args.trials)
            for trial in range(args.trials):
                logger.info("Trial #" + str(args.seed + trial + 1) + ":")

                random.seed(args.seed + trial)
                np.random.seed(args.seed + trial)
                torch.manual_seed(args.seed + trial)

                if args.split:
                    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = util.split(X, y)
                else:
                    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = X, y, None, None, None, None

                n, d = Xtrain.shape

                # Default regularization parameters
                if args.reg is not None:
                    C = args.reg
                elif mode[:7] == "sklearn":
                    if mode == "sklearnnb":
                        C = [None]
                    elif mode == "sklearnsgd":
                        C = [0] + list(map(lambda x: 10 ** x, range(-5, 6)))
                    else:
                        C = list(map(lambda x: 10 ** x, range(-5, 6))) + [10 ** 10]
                else:
                    C = [0] + list(map(lambda x: 10 ** x, range(-5, 6)))

                # Default choices of k
                if args.k is not None:
                    k = args.k
                else:
                    if mode in ["atk", "top", "border", "close", "close_decay", "drop"]:
                        # k = range(n + 1)
                        k = ([10 ** i for i in range(len(str(n - 1)))] + [n])
                    else:
                        k = [None]

                hyperparameters = list(itertools.product(C, k))

                best = None
                for hp in hyperparameters:
                    C, k = hp

                    if mode[:7] == "sklearn":
                        if mode == "sklearnlogistic":
                            model = sklearn.linear_model.LogisticRegression(
                                C=C, random_state=args.seed)
                        elif mode == "sklearnsvm":
                            model = sklearn.svm.LinearSVC(C=C, random_state=args.seed)
                        elif mode == "sklearnnb":
                            model = sklearn.naive_bayes.GaussianNB()
                        elif mode == "sklearnsgd":
                            model = sklearn.linear_model.SGDClassifier(
                                loss=args.loss if args.loss is not "logistic" else "log",
                                max_iter=args.epochs)
                        else:
                            raise ValueError("mode " + mode + " is not recognized")
                        model.fit(Xtrain, ytrain)
                        res = {}
                        res["train"] = sum(ytrain == model.predict(Xtrain)) / ytrain.size
                        res["valid"] = sum(yvalid == model.predict(Xvalid)) / yvalid.size
                        res["test"] = sum(ytest == model.predict(Xtest)) / ytest.size
                    else:
                        if args.pretrain and args.model != "linear":
                            logger.warning("Pretraining is only compatible with linear models.")

                        if args.model == "linear":
                            if args.pretrain:
                                model = util.linear_regression(Xtrain, ytrain, bias=True)
                            else:
                                model = torch.nn.Linear(d, 1)
                        elif args.model == "nn":
                            model = torch.nn.Sequential(
                                torch.nn.Linear(d, 2 * d), torch.nn.ReLU(),
                                torch.nn.Linear(2 * d, 1))
                        elif args.model == "dense":
                            model = dense.DenseBlock(2, d)
                        else:
                            raise ValueError("model " + args.model + " is not recognized")

                        model = torch.nn.Sequential(model)

                        res = train(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, model, loss,
                                    mode=mode, threshold=args.threshold, k=k,
                                    batch=args.batch, lr=args.lr, epochs=args.epochs, C=C,
                                    tol=args.tol, gpu=args.gpu)
                    if args.save:
                        pathlib.Path(args.logfile + "_checkpoints").mkdir(parents=True,
                                                                          exist_ok=True)
                        with open(args.logfile + "_checkpoints/" +
                                  (mode if len(args.mode) != 1 else "") +
                                  "_".join([str(args.seed + trial)] + list(map(str, hp))) +
                                  ".pkl", "wb") as f:
                            pickle.dump(model, f)

                    if best is None or ((res["valid"] >= best["valid"])
                                        if ("valid" in res)
                                        else (res["train"] >= best["train"])):
                        best = res
                        best_model = model
                    logger.info("        " + str(hp) + ": " + str(res))
                accuracy[trial] = best["test"] if "test" in best else best["train"]
                logger.info("    Accuracy: %f", accuracy[trial])
                logger.info("    Average Accuracy: %f (%f)",
                            accuracy[:(trial + 1)].mean(), accuracy[:(trial + 1)].std())
                logger.info("    Average Error: %f", 1 - accuracy[:(trial + 1)].mean())
                solution[m] = (mode, best_model)

        if args.render:
            pathlib.Path("fig/").mkdir(parents=True, exist_ok=True)
            if args.labels is None:
                args.labels = args.mode
            util.render(X, y, solution, "fig/" + args.dataset, X_remaining,
                        labels=args.labels)

    except Exception as e:
        logger.exception(traceback.format_exc())
        raise


def train(Xtrain, ytrain,
          Xvalid, yvalid,
          Xtest, ytest,
          model, loss,
          epochs=100,
          lr=1e-2,
          momentum=0.9,
          batch=1,
          mode="average",
          threshold=None,
          k=1,
          C=0,
          tol=None,
          gpu=False):
    """
    Trains a model using SGD with momentum.

    This function is used to create Figure 1.

    Args:
        Xtrain (np.array): training features
        ytrain (np.array): training classes
        Xvalid (np.array): validation features
        yvalid (np.array): validation classes
        Xtest (np.array): test features
        ytest (np.array): test classes
        model (torch.nn.Module): initial model
        epochs (int): maximum number of epochs
        lr (float): learning rate
        momentum (float): momentum for SGD
        batch (int): batch size
        mode (str): aggregate loss to train with
        threshold (float): threshold for close-k
                           (individual loss corresponding to correct prediction)
        k (int): number of examples for atk, close-k, and top-k
        C (float): regularization parameter
        tol (float): tolerance for stopping (set to None to train for all epochs)
        gpu (bool) train on GPU

    Returns:
        dict: training, validation, and test accuracies
            maps from string ("train", "valid", or "test") to the corresponding accuracy (float)
    """


    # Converting types and shape of data for downstream use
    Xtrain = torch.Tensor(Xtrain)
    ytrain = torch.Tensor(2 * ytrain - 1)
    if len(ytrain.shape) == 1:
        ytrain = ytrain.unsqueeze(1)

    if Xvalid is not None:
        Xvalid = torch.Tensor(Xvalid)
        yvalid = torch.Tensor(2 * yvalid - 1)
        if len(yvalid.shape) == 1:
            yvalid = yvalid.unsqueeze(1)

    if Xtest is not None:
        Xtest = torch.Tensor(Xtest)
        ytest = torch.Tensor(2 * ytest - 1)
        if len(ytest.shape) == 1:
            ytest = ytest.unsqueeze(1)

    if gpu:
        # Move data to GPU
        model.cuda()
        Xtrain = Xtrain.cuda()
        ytrain = ytrain.cuda()
        if Xvalid is not None:
            Xvalid = Xvalid.cuda()
            yvalid = yvalid.cuda()
        if Xtest is not None:
            Xtest = Xtest.cuda()
            ytest = ytest.cuda()

    logger = logging.getLogger(__name__)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=C)

    res = {}
    stop = False
    if tol is not None:
        prev_param = [p.clone() for p in model.parameters()]

    last_reset = 0
    best_agg = math.inf
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = -1
    for epoch in range(epochs):
        print_diagnostics = (epoch - last_reset < 5 or
                             all(map(lambda x: x == "9", str(epoch))) or
                             stop)

        prev_model = copy.deepcopy(model.state_dict())

        if print_diagnostics:
            logger.debug("Epoch #" + str(epoch + 1) + ":")

        run_test = Xtest is not None and (stop or epoch + 1 == epochs)
        for (dataset, X, y) in ([("train", Xtrain, ytrain)] +
                                ([("valid", Xvalid, yvalid)] if Xvalid is not None else []) +
                                ([("test", Xtest, ytest)] if run_test else [])):

            torch.set_grad_enabled(dataset == "train")

            total = 0.
            agg = 0.
            correct = 0
            prob = 0.

            n = X.shape[0]
            if mode == "average":
                for start in range(0, n, batch):
                    end = min(n, start + batch)
                    z = model(X[start:end, :])
                    pred = 2 * (z >= 0).type(torch.float) - 1
                    l = torch.sum(loss(z, y[start:end]))

                    prob += torch.sum((1. / (1 + torch.exp(-y[start:end] * z)))).cpu().detach().numpy()
                    correct += torch.sum(pred == y[start:end]).cpu().numpy()
                    total += l.cpu().detach().numpy()
                    agg += l.cpu().detach().numpy()

                    if dataset == "train" and epoch != 0:
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
            elif mode == "atk" or mode == "top" or mode == "maximal" or mode == "border" or mode == "close" or mode == "close_decay":
                l = torch.empty(n, 1)
                for start in range(0, n, batch):
                    end = min(n, start + batch)
                    z = model(X[start:end, :])
                    pred = 2 * (z >= 0).type(torch.float) - 1
                    l[start:end] = loss(z, y[start:end])

                    prob += torch.sum((1. / (1 + torch.exp(-y[start:end] * z)))).cpu().detach().numpy()
                    correct += torch.sum(pred == y[start:end]).cpu().numpy()
                total = torch.sum(l).cpu().detach().numpy()

                if dataset == "train":
                    M = 1
                    l, perm = torch.sort(l, dim=0, descending=True)
                    p, _ = torch.sort(pred == y[start:end], dim=0, descending=False)
                    if mode == "atk":
                        l = torch.sum(l[:k])
                    elif mode == "top":
                        l = torch.sum(l[k - 1])
                    elif mode == "maximal":
                        l = torch.sum(l[0])
                    elif mode == "border":
                        l = torch.sum(l[max(0, n - correct - k):(n - correct + k)])
                        l += M * correct
                    elif mode == "close" or mode == "close_decay":
                        if mode == "close":
                            k_ = k
                        else:
                            if epoch < epochs // 3:
                                k_ = n
                            elif epoch < 2 * epochs // 3:
                                k_ = k + round((n - k) * (2 * epochs // 3 - epoch) / epochs * 3)
                            else:
                                k_ = k
                        diff = torch.abs(l - threshold)
                        s, _ = torch.sort(diff, dim=0, descending=False)
                        ind = (((diff <= s[k_ - 1]) + torch.isnan(l)) != 0)
                        l = torch.sum(l[ind]) + M * (torch.sum((diff > s[k_ - 1]) * (l > threshold))).type(torch.float)
                    else:
                        raise ValueError("mode " + mode + " is not recognized")
                    agg = l.cpu().detach().numpy()
                    if epoch != 0:
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
            elif mode == "drop":
                l = torch.empty(n, 1)
                for start in range(0, n, batch):
                    end = min(n, start + batch)
                    z = model(X[start:end, :])
                    pred = 2 * (z >= 0).type(torch.float) - 1
                    l[start:end] = loss(z, y[start:end])

                    prob += torch.sum((1. / (1 + torch.exp(-y[start:end] * z)))).cpu().detach().numpy()
                    correct += torch.sum(pred == y[start:end]).cpu().numpy()
                total = torch.sum(l).cpu().detach().numpy()

                if dataset == "train":
                    if epoch < epochs // 3:
                        k_ = n
                    elif epoch < 2 * epochs // 3:
                        k_ = k + round((n - k) * (2 * epochs // 3 - epoch) / epochs * 3)
                    else:
                        k_ = k

                    if k_ < Xtrain.shape[0]:
                        diff = torch.abs(l - threshold)
                        a, perm = torch.sort(diff, dim=0, descending=False)
                        perm, _ = torch.sort(perm[:k_])
                        perm = perm.squeeze(1)
                        Xtrain = Xtrain[perm, :]
                        ytrain = ytrain[perm]

                    l = torch.sum(l)

                    agg = l.cpu().detach().numpy()
                    if epoch != 0:
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
            elif mode == "trunc":
                for start in range(0, n, batch):
                    end = min(n, start + batch)
                    z = model(X[start:end, :])
                    pred = 2 * (z >= 0).type(torch.float) - 1
                    l = torch.sum(loss(z, y[start:end]).clamp(max=2))

                    prob += torch.sum((1. / (1 + torch.exp(-y[start:end] * z)))).cpu().detach().numpy()
                    correct += torch.sum(pred == y[start:end]).cpu().numpy()
                    total += l.cpu().detach().numpy()
                    agg += l.cpu().detach().numpy()

                    if dataset == "train" and epoch != 0:
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
            else:
                raise NotImplementedError("Mode \"" + mode + " is not recognized.")

            res[dataset] = correct / float(n)

            if print_diagnostics:
                logger.debug("    %s:", dataset)
                logger.debug("        Cross-entropy: %f", total / float(n))
                if dataset == "train":
                    logger.debug("        Aggregate:     %f", agg / float(n))
                logger.debug("        Accuracy:      %f", correct / float(n))
                logger.debug("        Soft Accuracy: %f", prob / float(n))

            if dataset == "train":
                agg /= float(n)
                if math.isnan(total) or math.isnan(agg) or agg > 1.1 * best_agg:
                    lr /= 2
                    logger.debug("Resetting and dividing lr by 2 at epoch #%d (new lr=%f)", epoch + 1, lr)
                    logger.debug("Best model was epoch #%d with %f",best_epoch, best_agg)
                    last_reset = epoch
                    model.load_state_dict(best_model)
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=C)
                elif agg < best_agg:
                    best_epoch = epoch
                    best_agg = agg
                    best_model = prev_model

        if stop:
            break

        if lr < 1e-10:
            logger.debug("Learning rate too small at epoch #" + str(epoch))
            stop = True

        if tol is not None and last_reset + 5 <= epoch:
            delta = math.sqrt(sum([torch.norm(u - v) for (u, v) in zip(prev_param, model.parameters())]).detach().cpu().numpy()) 
            if delta < tol:
                logger.debug("Early termination at epoch #" + str(epoch) + " with change " + str(delta) + " < " + str(tol))
                stop = True
            prev_param = [p.clone() for p in model.parameters()]

    if gpu:
        model.cpu()

    return res


if __name__ == "__main__":
    main()
