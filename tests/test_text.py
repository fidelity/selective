# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import pandas as pd

from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest


class TestText(BaseTest):

    def test_text_based(self):
        data = pd.DataFrame({"item1": ["This is content 1"],
                             "item2": ["This is content 2"]})

        labels = pd.DataFrame({"item1": [0, 1, 1],
                               "item2": [1, 0, 1]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=1,
                                           featurization_method=None,
                                           optimization_method="exact",
                                           cost_metric="unicost")
        selector = Selective(method)

        selector.fit(data, labels)
        subset = selector.transform(data)

    def test_text_invalid_none_data(self):

        data = None
        labels = None

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=None,
                                           optimization_method="exact",
                                           cost_metric="diverse")
        selector = Selective(method)

        with self.assertRaises(ValueError):
            selector.fit(data, labels)

    def test_text_invalid_none_labels(self):
        data = pd.DataFrame()
        labels = None

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=None,
                                           optimization_method="exact",
                                           cost_metric="diverse")
        selector = Selective(method)

        with self.assertRaises(ValueError):
            selector.fit(data, labels)

    def test_text_invalid_opt_method(self):

        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=None,
                                               optimization_method="invalid",
                                               cost_metric="diverse")

            selector = Selective(method)

    def test_text_invalid_cost_metric(self):

        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=None,
                                               optimization_method="exact",
                                               cost_metric="invalid")

            selector = Selective(method)
