Close-k Classifier
==================

This repository contains code accompanying
> [Minimizing Close-k Aggregate Loss Improves Classification](https://arxiv.org/abs/1811.00521)
>
> Bryan He, James Zou.

We provide a Python 3 implementation using the
[scikit-learn API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects),
and provide code to reproduce the figures and tables from the paper.

Installation
------------

Our package is available on [PyPy](https://pypi.org/project/closek/), and can be installed using
> pip install -i https://pypi.org/project/ closek

You can also install this package by cloning the [Github repository](github.com/bryan-he/closek), and running
> pip install closek

If you want directly use the implementation in your package, you can also copy
[closek/closek.py](github.com/bryan-he/closekclosek/closek.py) into your code.

Usage
-----

An example of how to use our package is shown in [test.py](github.com/bryan-he/closek/test.py).

Generating Results from Paper
-----------------------------

The code used for the paper is in [experiments](github.com/bryan-he/closek/experiments). See the README
there for more details.

