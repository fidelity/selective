# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3


from sklearn.datasets import fetch_california_housing, load_iris
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestVIF(BaseTest):

    def test_vif_top_k_with_label(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=2, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['sepal width (cm)', 'petal width (cm)'])

    def test_vif_top_k_no_label(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=2, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['sepal width (cm)', 'petal width (cm)'])

    def test_vif_top_k_regression(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Statistical(num_features=2, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['HouseAge', 'AveOccup'])

    def test_vif_top_percentile(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=0.5, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['sepal width (cm)', 'petal width (cm)'])

    def test_vif_top_percentile_all(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.Statistical(num_features=1.0, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_vif_top_k_all(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=4, method="variance_inflation")
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
