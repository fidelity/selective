# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import pandas as pd

from feature.selector import Selective, SelectionMethod
from test_base import BaseTest
from textwiser import TextWiser, Embedding, Transformation


class TestText(BaseTest):

    def test_text_based_random(self):
        data = pd.DataFrame({"item1": ["This is content 1"],
                             "item2": ["This is content 2"],
                             "item3": ["This is content 3"]})
        labels = pd.DataFrame({"item1": [0, 1, 1], "item2": [1, 0, 1], "item3": [1, 1, 0]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=2,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="random")

        selector = Selective(method)
        selector.fit(data, labels)
        # subset = selector.transform(data)


    def test_text_based_greedy_uni(self):
        data = pd.DataFrame({"item1": ["This is content 1"],
                             "item2": ["This is content 2"],
                             "item3": ["This is content 3"],
                             "item4": ["This is content 4"],
                             "item5": ["This is content 5"],})
        labels = pd.DataFrame({"item1": [1, 0, 0, 1, 0], "item2": [0, 1, 1, 0, 0], "item3": [0, 0, 1, 1, 0],
                               "item4": [1, 0, 0, 0, 0], "item5": [0, 0, 0, 0, 1]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=2,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="greedy",
                                           cost_metric="diverse")
        selector = Selective(method)
        selector.fit(data, labels)
        #subset = selector.transform(data)



    def test_text_invalid_none_data(self):

        data = None
        labels = None

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="exact",
                                           cost_metric="diverse")
        selector = Selective(method)

        with self.assertRaises(ValueError):
            selector.fit(data, labels)

    def test_text_invalid_none_labels(self):
        data = pd.DataFrame()
        labels = None

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="exact",
                                           cost_metric="diverse")
        selector = Selective(method)

        with self.assertRaises(ValueError):
            selector.fit(data, labels)

    def test_text_invalid_opt_method(self):

        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=TextWiser(Embedding.TfIdf(),
                                                                              Transformation.NMF()),
                                               optimization_method="invalid",
                                               cost_metric="diverse")

            selector = Selective(method)

    def test_text_invalid_cost_metric(self):

        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=TextWiser(Embedding.TfIdf(),
                                                                              Transformation.NMF()),
                                               optimization_method="exact",
                                               cost_metric="invalid")

            selector = Selective(method)

    def test_text_invalid_featurization(self):
        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=None,
                                               optimization_method="exact",
                                               cost_metric="invalid")

            selector = Selective(method)
