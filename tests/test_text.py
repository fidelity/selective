# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import selectors
import numpy as np
import pandas as pd
from feature.selector import Selective, SelectionMethod
from test_base import BaseTest
from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions


class TestText(BaseTest):

    """
        For details about test parameters see TextBased in Selector.py
    """
    ################################################
    ########## Tests for random selection ##########
    ################################################
    def test_text_based_random_num_feature(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        method_unicost = SelectionMethod.TextBased(num_features=3,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=1)

        method_diverse = SelectionMethod.TextBased(num_features=3,
                                                   optimization_method="random",
                                                   cost_metric="diverse",
                                                   trials=1)

        selector_unicost = Selective(method_unicost)
        selector_unicost.fit(data, labels)
        selected_features_unicost = set(selector_unicost.transform(data).columns)

        selector_diverse = Selective(method_diverse)
        selector_diverse.fit(data, labels)
        selected_features_diverse = set(selector_diverse.transform(data).columns)

        assert selected_features_unicost == selected_features_diverse

    def test_text_based_random_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=5)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Check whether the selector.transform() is returned a DataFrame or not
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Check whether the number of row `selected_features' is the same as
        # the number of rows in the input data or not
        self.assertEqual(selected_features.shape[0], data.shape[0])

        # Check whether the number of columns in `selected_features' is equal to num_features (user input) or not
        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    def test_text_based_random_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        # "diverse" is Default cost metric
        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="random",
                                           trials=5)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        self.assertEqual(selected_features.shape[0], data.shape[0])

        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    ################################################
    ########## Tests for greedy selection ##########
    ################################################
    def test_text_based_greedy_num_feature_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0], "item3": [1, 0, 1, 0, 0],
                               "item4": [1, 0, 0, 0, 0], "item5": [0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=2,
                                           optimization_method="greedy",
                                           cost_metric="unicost")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        self.assertEqual(selected_features.shape[0], data.shape[0])

        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    def test_text_based_greedy_num_feature_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0], "item3": [1, 0, 1, 0, 0],
                               "item4": [1, 0, 0, 0, 0], "item5": [0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=2,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        self.assertEqual(selected_features.shape[0], data.shape[0])

        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    def test_text_based_greedy_unicost(self):
        data = pd.DataFrame({"item1": ["this is a sentences with more common words and more words to increase frequency"],
                             "item2": ["second one in list of more frequent sentences with some repeated words"],
                             "item3": ["a word for more complexity and longer length"],
                             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
                             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="greedy",
                                           cost_metric="unicost")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        self.assertEqual(selected_features.shape[0], data.shape[0])

        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    def test_text_based_greedy_diverse(self):
        data = pd.DataFrame({"item1": ["this is a sentences with more common words and more words to increase frequency"],
                             "item2": ["second one in list of more frequent sentences with some repeated words"],
                             "item3": ["a word for more complexity and longer length"],
                             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
                             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 1, 0], "item2": [0, 1, 0, 0, 0], "item3": [0, 0, 1, 0, 0],
                               "item4": [0, 0, 1, 0, 0], "item5": [0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        self.assertEqual(selected_features.shape[0], data.shape[0])

        if method.num_features is None:
            pass
        else:
            self.assertEqual(selected_features.shape[1], 2)

    ################################################
    ########## Verify invalid tests  ###############
    ################################################
    def test_text_invalid_none_data(self):

        data = None
        labels = None

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(),
                                                                          Transformation.NMF()),
                                           optimization_method="random",
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
                                           optimization_method="random",
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
                                               optimization_method="random",
                                               cost_metric="invalid")

            selector = Selective(method)

    def test_text_invalid_featurization(self):
        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=None,
                                               optimization_method="exact",
                                               cost_metric="invalid")

            selector = Selective(method)