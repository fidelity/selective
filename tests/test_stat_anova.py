# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from sklearn.datasets import fetch_california_housing, load_iris
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestANOVA(BaseTest):

    def test_anova_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Statistical(num_features=3)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveRooms'])

    def test_anova_regress_top_percentile(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Statistical(num_features=0.6)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveRooms'])

    def test_anova_regress_top_k_all(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Statistical(num_features=5)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], subset.shape[1])
        self.assertListEqual(list(data.columns), list(subset.columns))

    def test_anova_regress_top_percentile_all(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Statistical(num_features=1.0)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], subset.shape[1])
        self.assertListEqual(list(data.columns), list(subset.columns))

    def test_anova_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=2)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_anova_classif_top_percentile(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.Statistical(num_features=0.5)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_anova_classif_top_percentile_all(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=1.0)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_anova_classif_top_k_all(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.Statistical(num_features=4)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])