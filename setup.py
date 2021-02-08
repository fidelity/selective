# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open(os.path.join('feature', '_version.py')) as fp:
    exec(fp.read())

setuptools.setup(
    name="selective",
    description="feature selection library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author="FMR LLC",
    url="https://github.com/fidelity/selective",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={"Source": "https://github.com/fidelity/selective"},
    install_requires=required,
    python_requires=">=3.6"
)
