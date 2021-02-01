# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3


import unittest


# Test Directory
start_dir = '.'

# Test Loader
loader = unittest.TestLoader()

# Test Suite
suite = loader.discover(start_dir)

# Test Runner
runner = unittest.TextTestRunner()

# Run the Suite
runner.run(suite)
