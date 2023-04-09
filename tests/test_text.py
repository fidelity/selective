# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import pandas as pd

from feature.selector import Selective, SelectionMethod
from test_base import BaseTest
from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions




class TestText(BaseTest):

    def test_text_based_random_or_kmeans(self):
        data = pd.DataFrame({"item1": ["this is a sentences"],
                             "item2": ["second one in list of sentences"],
                             "item3": ["a word for complexity"],
                             "item4": ["sentence with a lot of repeated words"],
                             "item5": ["number of words with a valid vector"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0], "item2": [0, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 0],
                               "item4": [1, 0, 1, 0, 0], "item5": [0, 1, 0, 0, 1]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="random")


        selector = Selective(method)
        selector.fit(data, labels)
        subset = selector.transform(data)
        print(subset)


    # need to add a proper test to see distinction between diverse and unicost
    # diverse cost metric takes dummy for below test
    def test_text_based_greedy_or_exact(self):
        data = pd.DataFrame({"item1": ["this is a sentences"],
                             "item2": ["second one in list of sentences"],
                             "item3": ["a word for complexity"],
                             "item4": ["sentence with a lot of repeated words"],
                             "item5": ["number of words with a valid vector"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0], "item2": [0, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 0],
                               "item4": [1, 0, 1, 0, 0], "item5": [0, 1, 0, 0, 1]})
        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=1),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy",
                                           cost_metric="unicost")

        selector = Selective(method)
        selector.fit(data, labels)
        subset = selector.transform(data)
        print(subset)




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
