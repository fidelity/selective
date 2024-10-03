# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

# from sklearn.datasets import load_boston, load_iris
# from feature.utils import get_data_label
# from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestMaximalInfo(BaseTest):

    def test_maximal(self):
        pass

    # def test_maximal_regress_top_k(self):
    #     data, label = get_data_label(load_boston())
    #     data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])
    #
    #     method = SelectionMethod.Statistical(num_features=3, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 3)
    #     self.assertListEqual(list(subset.columns), ['CRIM', 'AGE', 'LSTAT'])
    #
    # def test_maximal_regress_top_percentile(self):
    #     data, label = get_data_label(load_boston())
    #     data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])
    #
    #     method = SelectionMethod.Statistical(num_features=0.6, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 3)
    #     self.assertListEqual(list(subset.columns), ['CRIM', 'AGE', 'LSTAT'])
    #
    # def test_maximal_regress_top_k_all(self):
    #     data, label = get_data_label(load_boston())
    #     data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])
    #
    #     method = SelectionMethod.Statistical(num_features=5, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(data.shape[1], subset.shape[1])
    #     self.assertListEqual(list(data.columns), list(subset.columns))
    #
    # def test_maximal_regress_top_percentile_all(self):
    #     data, label = get_data_label(load_boston())
    #     data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])
    #
    #     method = SelectionMethod.Statistical(num_features=1.0, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(data.shape[1], subset.shape[1])
    #     self.assertListEqual(list(data.columns), list(subset.columns))
    #
    # def test_maximal_classif_top_k(self):
    #     data, label = get_data_label(load_iris())
    #
    #     method = SelectionMethod.Statistical(num_features=2, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 2)
    #     self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])
    #
    # def test_maximal_classif_top_percentile(self):
    #     data, label = get_data_label(load_iris())
    #
    #     method = SelectionMethod.Statistical(num_features=0.5, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 2)
    #     self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])
    #
    # def test_maximal_classif_top_percentile_all(self):
    #     data, label = get_data_label(load_iris())
    #
    #     method = SelectionMethod.Statistical(num_features=1.0, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 4)
    #     self.assertListEqual(list(subset.columns),
    #                          ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    #
    # def test_maximal_classif_top_k_all(self):
    #     data, label = get_data_label(load_iris())
    #
    #     method = SelectionMethod.Statistical(num_features=4, method="maximal_info")
    #     selector = Selective(method)
    #     selector.fit(data, label)
    #     subset = selector.transform(data)
    #
    #     # Reduced columns
    #     self.assertEqual(subset.shape[1], 4)
    #     self.assertListEqual(list(subset.columns),
    #                          ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
