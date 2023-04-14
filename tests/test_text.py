# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import pandas as pd

from feature.selector import Selective, SelectionMethod
from test_base import BaseTest
from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions



class TestText(BaseTest):

    """
        optimization_method : random
        num_features:
            - t = maximum number of features defines by user. The cost_metric input argument should ignore
            - t = None: the number of feature computed by solving a set cover problem with cost metrics
                (unicost or diverse)
    """
    def test_text_based_random_max_num_feature(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=3,
                                           optimization_method="random")

        selector = Selective(method)
        selector.fit(data, labels)
        selector.transform(data)


    def test_text_based_random_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        print(data)
        print(labels)
        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="random",
                                           cost_metric="unicost")


        selector = Selective(method)
        selector.fit(data, labels)
        selector.transform(data)

    def test_text_based_random_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})
        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="random",
                                           cost_metric="diverse")

        selector = Selective(method)
        selector.fit(data, labels)
        selector.transform(data)


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

####################################################################
########################other tests#################################
####################################################################

    """
    def test_text_based_kmeans_or_max_cover(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0], "item3": [1, 0, 1, 0, 0],
                               "item4": [1, 0, 0, 0, 0], "item5": [0, 1, 0, 0, 1]})

        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="kmeans")


        selector = Selective(method)
        selector.fit(data, labels)
        selector.transform(data)


        # optimization_method : kmeans
        # featurization_method(required)
        # num_features:
        #     - integer: defines by user
        #     - None: the number of feature computed by solving a set cover problem using labels
        #     
        # optimization_method : max_cover
        # featurization_method(required)
        # num_features:
        #     - integer: defines by user

    def test_text_based_greedy_or_exact(self):
        data = pd.DataFrame({"item1": ["this is a sentences with more common words and more words to increase frequency"],
                             "item2": ["second one in list of more frequent sentences with some repeated words"],
                             "item3": ["a word for more complexity and longer length"],
                             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
                             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 1, 0], "item2": [0, 1, 0, 0, 0], "item3": [0, 0, 1, 0, 0],
                               "item4": [0, 0, 1, 0, 0], "item5": [0, 1, 0, 0, 1]})
        print(data)
        print(labels)

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy",
                                           cost_metric="unicost")

        selector = Selective(method)
        selector.fit(data, labels)
        selector.transform(data)


            # optimization_method : greedy
            # featurization_method(required)
            # num_features:
            #     - integer: defines by user
            #     - None: the number of feature computed by solving a set cover problem using labels
            # cost_metric: unicost or diverse
            # 
            # optimization_method : exact
            # featurization_method(required)
            # num_features:
            #     - None: algorithm defines/optimizes the number of features
            # cost_metric: unicost or diverse
    """