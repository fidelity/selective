# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import unittest


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