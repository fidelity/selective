# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from sklearn.datasets import load_boston, load_iris
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestChi(BaseTest):

    def test_chi_regress_invalid(self):
        data, label = get_data_label(load_boston())
        data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])

        method = SelectionMethod.Statistical(num_features=3, method="chi_square")
        selector = Selective(method)
        with self.assertRaises(TypeError):
            selector.fit(data, label)

    def test_chi_regress_top_percentile_invalid(self):
        data, label = get_data_label(load_boston())
        data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])

        method = SelectionMethod.Statistical(num_features=0.6, method="chi_square")
        selector = Selective(method)
        with self.assertRaises(TypeError):
            selector.fit(data, label)

    def test_chi_classif_top_k(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.Statistical(num_features=2, method="chi_square")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_chi_classif_top_percentile(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.Statistical(num_features=0.5, method="chi_square")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_chi_classif_top_percentile_all(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.Statistical(num_features=1.0, method="chi_square")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_chi_classif_top_k_all(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=4, method="chi_square")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
