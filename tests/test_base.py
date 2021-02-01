# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import unittest
from sklearn.datasets import load_iris
from feature.utils import get_data_label, reduce_memory, DataTransformer


class BaseTest(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2):
        """
        Asserts that floating values in the given lists (almost) equals to each other
        """
        if not isinstance(list1, list):
            list1 = list(list1)

        if not isinstance(list2, list):
            list2 = list(list2)

        self.assertEqual(len(list1), len(list2))

        for index, val in enumerate(list1):
            self.assertAlmostEqual(val, list2[index], delta=0.01)

    @staticmethod
    def test_mem_usage():
        data, label = get_data_label(load_iris())
        data_reduced = reduce_memory(data, verbose=False)

    @staticmethod
    def test_cap_floor():
        data, label = get_data_label(load_iris())

        # Fit transformer and transform to numeric contexts
        data_transformer = DataTransformer()
        contexts = data_transformer.fit(data)
        contexts = data_transformer.transform(data)
        contexts = data_transformer.fit_transform(data)
