# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3


from sklearn.datasets import load_boston
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestVariance(BaseTest):

    def test_variance_no_threshold(self):
        data, label = get_data_label(load_boston())

        method = SelectionMethod.Variance()
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], 13)
        self.assertEqual(subset.shape[1], 13)

    def test_variance_zero_threshold(self):
        data, label = get_data_label(load_boston())

        method = SelectionMethod.Variance(threshold=0)
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], 13)
        self.assertEqual(subset.shape[1], 13)

    def test_variance_lt1(self):
        data, label = get_data_label(load_boston())

        method = SelectionMethod.Variance(threshold=1.0)
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], 13)
        self.assertEqual(subset.shape[1], 10)
        self.assertListEqual(list(subset.columns),
                             ['CRIM', 'ZN', 'INDUS', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    def test_variance_drop_target(self):
        data, label = get_data_label(load_boston())

        method = SelectionMethod.Variance(threshold=85)
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], 13)
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns), ['ZN', 'AGE', 'TAX', 'B'])

    def test_variance_drop_all(self):
        data, label = get_data_label(load_boston())
        method = SelectionMethod.Variance(threshold=100000)
        selector = Selective(method)
        try:
            selector.fit(data)
            selector.transform(data)
        except ValueError:
            pass

    def test_variance_lt1_fit_trans(self):
        data, label = get_data_label(load_boston())

        method = SelectionMethod.Variance(threshold=1.0)
        selector = Selective(method)
        subset = selector.fit_transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], 13)
        self.assertEqual(subset.shape[1], 10)
        self.assertListEqual(list(subset.columns),
                             ['CRIM', 'ZN', 'INDUS', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
