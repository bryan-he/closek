"""Scikit-learn-style implementation of the close-k classifier."""

import math
import copy

import numpy as np
import torch
import sklearn

class CloseKClassifier(sklearn.base.BaseEstimator):
    """
    Implementation of a classifier using the close-k aggregate loss.

    Args:
        loss (str): loss function to be used
            Possible options are "hinge" and "log"
            Defaults to "hinge"
        k (int): number of examples to use
        model (torch.nn.Module): type of model to train with
            Defaults to None, which automatically creates a torch.nn.Linear
            layer with the appropriate dimension
        decay (bool): whether or not to decay k
        alpha (float): weight decay parameter
        max_iter (int): number of epochs to run
        eta0 (float): learning rate
        momentum (float): momentum for SGD
    """
    def __init__(self, loss="hinge", k=None, model=None, decay=True,
                 alpha=0.0001, max_iter=1000, eta0=1e-1, momentum=0.9):
        self.loss = loss
        self.k = k
        self.model = model
        self.decay = decay
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0
        self.momentum = momentum

        self.fitted = None

    def fit(self, X, y):
        """
        Fits model to the given training data.

        No value is returned; the model is stored internally as `fitted`.

        Args:
            X (np.array, shape = (n_samples, n_features)): Training features
            y (np.array, shape = (n_samples) or (n_samples, n_outputs)): Training labels
        """
        n, d = X.shape
        if self.model is None:
            # Default to linear model
            self.fitted = torch.nn.Linear(d, 1)
        else:
            # Make fresh copy of model
            self.fitted = copy.deepcopy(model)

        k = self.k
        if k is None:
            k = n
        if k > n:
            k = n

        # Set loss function
        if self.loss == "hinge":
            loss = torch.nn.SoftMarginLoss(reduction="none")
            threshold = 1.
        elif self.loss == "log":
            loss = HingeLoss()
            threshold = math.log(2.)
        else:
            raise ValueError("Loss \"" + loss + "\" is not recognized.")

        # Converting types and shape of data for downstream use
        X = torch.Tensor(X)
        y = torch.Tensor(2 * y - 1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        optimizer = torch.optim.SGD(self.fitted.parameters(),
                                    lr=self.eta0 / n,
                                    momentum=self.momentum,
                                    weight_decay=self.alpha)

        for epoch in range(self.max_iter):
            z = self.fitted(X)
            l = loss(z, y)

            total = torch.sum(l).cpu().detach().numpy()
            if self.decay:
                if epoch < self.max_iter // 3:
                    k_ = n
                elif epoch < 2 * self.max_iter // 3:
                    k_ = k + round((n - k) * (2 * self.max_iter // 3 - epoch) / self.max_iter * 3)
                else:
                    k_ = k
            else:
                k = self.k

            diff = torch.abs(l - threshold)
            s, _ = torch.sort(diff, dim=0, descending=False)
            ind = (diff <= s[k_ - 1])
            l = torch.sum(l[ind])

            optimizer.zero_grad()
            l.backward()
            optimizer.step()


    def predict(self, X):
        """
        Predicts labels for test data.

        Args:
            X (np.array, shape = (n_samples, n_features)): Test features

        Returns
            np.array, shape = (n_samples) or (n_samples, n_outputs): Prediction for labels
        """
        if self.fitted is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                     "yet" % {'name': type(self).__name__})

        return (self.fitted(torch.Tensor(X)).detach().numpy().squeeze(1) > 0).astype(np.int64)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (np.array, shape = (n_samples, n_features)): Test features
            y (np.array, shape = (n_samples) or (n_samples, n_outputs)): Test labels

        Returns:
        -------
            float: Accuracy on test set
        """
        return np.mean(self.predict(X) == y)

    def decision_function(self, X):
        """
        Predicts confidence scores for test data.

        Args:
            X (np.array, shape = (n_samples, n_features)): Test features

        Returns
            np.array, shape = (n_samples) or (n_samples, n_outputs): Confidence score for test data
        """
        return self.fitted(torch.Tensor(X)).detach().numpy()


class HingeLoss(torch.nn.Module):
    """Neural network module for computing hinge loss."""
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pred, target):
        return (1 - pred * target).clamp(0)
