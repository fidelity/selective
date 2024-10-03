# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from sklearn.datasets import fetch_california_housing
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestCorrelation(BaseTest):

    def test_correlation_small_kendall(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Correlation(0.60, method="kendall")
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup'])

    def test_correlation_small_spearman(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Correlation(0.60, method="spearman")
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveBedrms', 'AveOccup'])

    def test_correlation_invalid_method(self):
        with self.assertRaises(ValueError):
            method = SelectionMethod.Correlation(0.60, method="xx")
            selector = Selective(method)

    def test_correlation_small_pearson(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Correlation(0.60, method="pearson")
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup'])

    def test_correlation_small(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.Correlation(0.60)
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup'])

    def test_correlation(self):
        data, label = get_data_label(fetch_california_housing())

        method = SelectionMethod.Correlation(0.60)
        selector = Selective(method)
        selector.fit(data)
        subset = selector.transform(data)

        self.assertListEqual(list(subset.columns),
                             ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Latitude'])

    def test_correlation_fit_trans(self):
        data, label = get_data_label(fetch_california_housing())

        method = SelectionMethod.Correlation(0.60)
        selector = Selective(method)
        subset = selector.fit_transform(data)

        self.assertListEqual(list(subset.columns),
                             ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Latitude'])