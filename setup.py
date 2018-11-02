"""Metadata for package to allow installation with pip."""

import setuptools

exec(open("closek/version.py").read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="closek",
    description="Scikit-learn-style implementation of the close-k classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bryan He",
    author_email="bryanhe@stanford.edu",
    url="https://github.com/bryan-he/close-k",
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "sklearn",
    ],
    tests_require=[
        "pmlb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
