#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Wrapper for pyqg experiments"
version="0.0.1"

setup(name="pyqg_explorer",
    version=version,
    description=description,
    url="https://github.com/Chris-Pedersen/pyqg_explorer",
    author="Chris Pedersen",
    author_email="c.pedersen@nyu.edu",
    packages=find_packages(),
    )
