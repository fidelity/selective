# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import numpy as np
import pandas as pd
import random
import sklearn
from feature.selector import Selective, SelectionMethod
from tests.test_base import BaseTest
from textwiser import TextWiser, Embedding, Transformation


class TestText(BaseTest):

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

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Fic the random seed
        seed = 123
        np.random.seed(seed)

        method_unicost = SelectionMethod.TextBased(num_features=3,
                                                   optimization_method="random",
                                                   cost_metric="unicost",
                                                   trials=1)

        method_diverse = SelectionMethod.TextBased(num_features=3,
                                                   optimization_method="random",
                                                   cost_metric="diverse",
                                                   trials=1)

        selector_unicost = Selective(method_unicost, seed=seed)
        selector_unicost.fit(data, labels)
        selected_features_unicost = selector_unicost.transform(data)

        # Set the same seed again
        np.random.seed(seed)
        selector_diverse = Selective(method_diverse, seed=seed)
        selector_diverse.fit(data, labels)
        selected_features_diverse = selector_diverse.transform(data)

        # Check whether the seed value used in tests is the same (not the default = 123456)
        self.assertEqual(selector_unicost.seed, selector_diverse.seed)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features_unicost.equals(selected_features_diverse))

        # Find the best solution using multiple trails and assert that it is selected with the same seed
        method = SelectionMethod.TextBased(num_features=3,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=20)

        # Set the same seed again
        np.random.seed(seed)
        best_selector = Selective(method, seed=seed)
        best_selector.fit(data, labels)
        best_selected_features = best_selector.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features_unicost.equals(best_selected_features))

        # Verify the selected indices
        self.assertListEqual(list(best_selector.transform(data).columns), ['item2', 'item3', 'item5'])

    def test_text_based_random_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Use same seed for both methods
        seed = 123
        np.random.seed(seed)

        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=5)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Check whether the seed value used in test is the same (not the default = 123456)
        self.assertEqual(selector.seed, seed)

        # Check whether the selector.transform() is returned a DataFrame or not
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Check whether the number of columns in `selected_features' is equal to num_features selected
        # by solving set cover
        self.assertEqual(selected_features.shape[1], 2)

        # Verify the selected indices
        self.assertListEqual(list(selected_features.columns), ['item2', 'item3'])
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

    def test_text_based_random_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})
        labels = pd.DataFrame({"item1": [0, 1, 0, 0, 0], "item2": [1, 0, 0, 1, 0], "item3": [0, 0, 1, 0, 1],
                               "item4": [0, 1, 1, 0, 1], "item5": [0, 1, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        seed = 123
        np.random.seed(seed)

        # "diverse" is Default cost metric
        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="random",
                                           cost_metric="diverse",
                                           trials=5)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Check whether the seed value used in test is the same (not the default = 123456)
        self.assertEqual(selector.seed, seed)

        # Check whether the selector.transform() is returned a DataFrame or not
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Check whether the number of columns in `selected_features' is equal to num_features selected
        # by solving set cover
        self.assertEqual(selected_features.shape[1], 2)

        # Verify the selected indices
        self.assertListEqual(list(selected_features.columns), ['item2', 'item3'])
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

    ################################################
    ########## Tests for greedy selection ##########
    ################################################
    def test_text_based_greedy_num_feature_one_or_infeasible_or_max(self):
        # Fix the random seed
        seed = 12
        np.random.seed(seed)

        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})

        ### Labels with a single column with all ones ###
        labels_single_col = pd.DataFrame({"item1": [1, 1, 1, 1, 1], "item2": [0, 0, 0, 0, 0], "item3": [0, 0, 0, 0, 0],
                                          "item4": [0, 0, 0, 0, 0], "item5": [0, 0, 0, 0, 0]})

        method_single_col = SelectionMethod.TextBased(num_features=1,
                                                      optimization_method="greedy",
                                                      cost_metric="unicost")

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels_single_col.shape[1])

        selector = Selective(method_single_col, seed=seed)
        selector.fit(data, labels_single_col)
        selected_features_single_col = selector.transform(data)

        assert selector.selection_method.trials == 10
        self.assertTrue(isinstance(selected_features_single_col, pd.DataFrame))

        # Verify greedy selects the item1
        self.assertListEqual(list(selected_features_single_col.columns), ["item1"])

        ### Labels for infeasible instance ###
        labels_infeasible = pd.DataFrame({"item1": [0, 0, 0, 0, 0], "item2": [0, 0, 0, 0, 0], "item3": [0, 0, 0, 0, 0],
                                          "item4": [0, 0, 0, 0, 0], "item5": [0, 0, 0, 0, 0]})

        method_infeasible = SelectionMethod.TextBased(num_features=2,
                                                      optimization_method="greedy",
                                                      cost_metric="unicost")

        # Set the same seed again
        np.random.seed(seed)
        selector = Selective(method_infeasible, seed=seed)
        selector.fit(data, labels_infeasible)
        selected_features_infeasible = selector.transform(data)

        assert selector.selection_method.trials == 10
        self.assertTrue(isinstance(selected_features_infeasible, pd.DataFrame))

        # Verify that there is at least one selected column for each label
        feasible = True
        for label in labels_infeasible.columns:
            if label not in selected_features_infeasible.columns:
                feasible = False
                break

        # Check whether feasible is False
        self.assertFalse(feasible)

        ### Labels with a column with considerable coverage ###
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 1], "item3": [1, 0, 1, 0, 1],
                               "item4": [1, 0, 0, 0, 0], "item5": [0, 1, 0, 0, 0]})

        method_max_cols = SelectionMethod.TextBased(num_features=2,
                                                    optimization_method="greedy",
                                                    cost_metric="unicost")

        # Set the same seed again
        np.random.seed(seed)
        selector = Selective(method_max_cols, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10
        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 2)

        # Verify selected features with considerable label coverage
        self.assertListEqual(list(selected_features.columns), ['item2', 'item3'])

    def test_text_based_greedy_num_feature_unicost_diverse(self):
        ### Several highly correlated features but grouped based on labels ###
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["third sentence with repeated words as item2"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["two words for more complexity and longer length"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 0, 0, 1, 0, 0, 0],
                               "item3": [1, 0, 1, 0, 0, 1, 0], "item4": [0, 1, 0, 0, 0, 0, 0],
                               "item5": [0, 1, 0, 0, 1, 0, 0], "item6": [1, 0, 0, 0, 0, 0, 0],
                               "item7": [1, 0, 0, 1, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Fix the random seed due to use highly correlated features and randomness in the Lagrangian multiplier
        seed = 12
        np.random.seed(seed)

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy",
                                           cost_metric="diverse")

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 10
        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 3)

        # Verify that features are selected are low correlated
        self.assertListEqual(list(selected_features.columns), ['item3', 'item5', 'item7'])

    def test_text_based_greedy_unicost_diverse_identity(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["third sentence with repeated words as item2"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["two words for more complexity and longer length"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})

        ### Labels with an identity columns ###
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 1, 0, 0, 0, 0, 0],
                               "item3": [0, 0, 1, 0, 0, 0, 0], "item4": [0, 0, 0, 1, 0, 0, 0],
                               "item5": [0, 0, 0, 0, 1, 0, 0], "item6": [0, 0, 0, 0, 0, 1, 0],
                               "item7": [0, 0, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Fix the random seed
        seed = 12
        np.random.seed(seed)

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="greedy",
                                           cost_metric="diverse")

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)
        print(selected_features)

        assert selector.selection_method.trials == 10
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Verify all features are selected
        self.assertListEqual(selected_features.columns.tolist(), data.columns.tolist())

    ################################################
    ########## Tests for kmeans selection ##########
    ################################################
    def test_text_based_kmeans_num_feature(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["third sentence with repeated words as item2"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["two words for more complexity and longer length"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 0, 0, 1, 0, 0, 0],
                               "item3": [1, 0, 1, 0, 0, 1, 0], "item4": [0, 1, 0, 0, 0, 0, 0],
                               "item5": [0, 1, 0, 0, 1, 0, 0], "item6": [1, 0, 0, 0, 0, 0, 0],
                               "item7": [1, 0, 0, 1, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=2,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="kmeans",
                                           cost_metric="unicost")  # Default is diverse

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Set random seed for both NumPy and scikit-learn
        seed = 1234
        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertEqual(selector.seed, 1234)
        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 2)

        # Check the selected features are a subset of the original columns
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

        # Verify that features are selected are low correlated
        self.assertListEqual(list(selected_features.columns), ['item1', 'item7'])

    def test_text_based_kmeans_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 1, 0, 0, 0, 0, 0],
                               "item3": [0, 0, 1, 0, 0, 0, 0], "item4": [0, 0, 0, 1, 0, 0, 0],
                               "item5": [1, 0, 0, 0, 1, 0, 0], "item6": [1, 0, 0, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=30),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="kmeans",
                                           cost_metric="unicost")  # Default cost metric is diverse

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Set random seed for both NumPy and scikit-learn
        seed = 1234
        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Verify that the features selected
        self.assertListEqual(list(selected_features.columns), ['item1', 'item2', 'item4', 'item5', 'item7'])

    def test_text_based_kmeans_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 1, 0, 0, 0, 0, 0],
                               "item3": [0, 0, 1, 0, 0, 0, 0], "item4": [0, 0, 0, 1, 0, 0, 0],
                               "item5": [1, 0, 0, 0, 1, 0, 0], "item6": [1, 0, 0, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=20),
                                                                           Transformation.SVD(n_components=10)]),
                                           optimization_method="kmeans",
                                           cost_metric="diverse")  # Default cost metric is diverse

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        # Set random seed for both NumPy and scikit-learn
        seed = 1234
        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Verify the consistency of selected features with the initial run
        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)
        selector2 = Selective(method, seed=seed)

        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        method2 = SelectionMethod.TextBased(num_features=None,
                                            featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                           [Transformation.NMF(n_components=30),
                                                                            Transformation.SVD(n_components=10)]),
                                            optimization_method="kmeans")
        selector3 = Selective(method2, seed=seed)
        selector3.fit(data, labels)
        selected_features3 = selector3.transform(data)

        # Verify that changing TextWiser parameters results in different selections
        self.assertNotEqual(selected_features.columns.tolist(), selected_features3.columns.tolist())

    ###############################################
    ########## Tests for exact selection ##########
    ###############################################
    def test_text_based_exact_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 1, 0, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0, 0, 1],
                               "item3": [0, 0, 1, 0, 1, 1, 0], "item4": [1, 0, 0, 1, 0, 1, 0],
                               "item5": [1, 0, 1, 0, 1, 0, 0], "item6": [1, 0, 1, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="exact",
                                           cost_metric="unicost",
                                           trials=1)  # Default is diverse

        method2 = SelectionMethod.TextBased(num_features=None,
                                            optimization_method="exact",
                                            cost_metric="unicost",
                                            trials=20)  # Default for trials=10

        # Set a fixed seed for the random number generator
        seed = 12345
        np.random.seed(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 1  # Only run once
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        np.random.seed(seed)  # Set the same seed again

        selector2 = Selective(method2, seed=seed)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item1', 'item2', 'item3'])

    def test_text_based_exact_num_feature_unicost(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 1, 0, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0, 0, 1],
                               "item3": [0, 0, 1, 0, 1, 1, 0], "item4": [1, 0, 0, 1, 0, 1, 0],
                               "item5": [1, 0, 1, 0, 1, 0, 0], "item6": [1, 0, 1, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=2,  # num_features is less than the solution of set cover
                                           optimization_method="exact",
                                           cost_metric="unicost",
                                           trials=1)  # Default is diverse

        method2 = SelectionMethod.TextBased(num_features=2,
                                            optimization_method="exact",
                                            cost_metric="unicost",
                                            trials=20)  # Default for trials=10

        # Set a fixed seed for the random number generator
        seed = 12345
        np.random.seed(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)
        print(selected_features.columns)

        assert selector.selection_method.trials == 1  # Only run once
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        np.random.seed(seed)  # Set the same seed again
        selector2 = Selective(method2, seed=seed)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item1', 'item3'])

    def test_text_based_exact_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 1, 0, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0, 0, 1],
                               "item3": [0, 0, 1, 0, 1, 1, 0], "item4": [1, 0, 0, 1, 0, 1, 0],
                               "item5": [1, 0, 1, 0, 1, 0, 0], "item6": [1, 0, 1, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=70),
                                                                           Transformation.SVD(n_components=20)]),
                                           optimization_method="exact",
                                           cost_metric="diverse",
                                           trials=1)  # Default cost metric is diverse

        method2 = SelectionMethod.TextBased(num_features=None,
                                            featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=70),
                                                                           Transformation.SVD(n_components=20)]),
                                            optimization_method="exact",
                                            cost_metric="diverse",
                                            trials=20)  # Default for trials=10

        # Set a fixed seed for the random number generator
        seed = 12345
        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)


        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 1  # Only run once
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        np.random.seed(seed)
        random.seed(seed)
        sklearn.set_config(working_memory=0)
        sklearn.utils.check_random_state(seed)  # Set the same seed again

        selector2 = Selective(method2, seed=seed)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item2', 'item5', 'item6'])

    def test_text_based_exact_num_feature_diverse(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})
        labels = pd.DataFrame({"item1": [1, 1, 0, 0, 0, 0, 1], "item2": [0, 1, 0, 1, 0, 0, 1],
                               "item3": [0, 0, 1, 0, 1, 1, 0], "item4": [1, 0, 0, 1, 0, 1, 0],
                               "item5": [1, 0, 1, 0, 1, 0, 0], "item6": [1, 0, 1, 0, 0, 1, 0],
                               "item7": [0, 1, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=2,  # num_features is less than the solution of set cover
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                          [Transformation.NMF(n_components=70),
                                                                           Transformation.SVD(n_components=20)]),
                                           optimization_method="exact",
                                           cost_metric="diverse",
                                           trials=1)  # Default cost metric is diverse

        method2 = SelectionMethod.TextBased(num_features=2,
                                            featurization_method=TextWiser(Embedding.TfIdf(min_df=0),
                                                                           [Transformation.NMF(n_components=70),
                                                                            Transformation.SVD(n_components=20)]),
                                            optimization_method="exact",
                                            cost_metric="diverse",
                                            trials=20)  # Default for trials=10

        # Set a fixed seed for the random number generator
        seed = 12345
        np.random.seed(seed)

        selector = Selective(method, seed=seed)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        assert selector.selection_method.trials == 1  # Only run once
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        np.random.seed(seed)  # Set the same seed again
        selector2 = Selective(method2, seed=seed)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item2', 'item6'])

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
                                               optimization_method="exact",
                                               cost_metric="invalid")

            selector = Selective(method)
