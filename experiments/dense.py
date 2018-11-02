"""Classes for neural network with dense connections."""

import torch

class DenseBlock(torch.nn.Sequential):
    """Sequence of hidden layers that reuse the original features."""
    def __init__(self, num_layers, ninput):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer((i + 1) * ninput, ninput)
            self.add_module("denselayer%d" % (i + 1), layer)
        self.add_module("classifier", torch.nn.Linear((num_layers + 1) * ninput, 1))

class DenseLayer(torch.nn.Sequential):
    """Layer in a provides a hidden layer and the original features."""
    def __init__(self, ninput, noutput):
        super(DenseLayer, self).__init__()
        self.add_module("linear", torch.nn.Linear(ninput, noutput)),
        self.add_module("relu", torch.nn.ReLU())

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)
