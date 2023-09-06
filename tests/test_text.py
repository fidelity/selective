# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3
import numpy as np
import pandas as pd
from feature.selector import Selective, SelectionMethod
from feature.text_based import process_category_data
from tests.test_base import BaseTest
from textwiser import TextWiser, Embedding, Transformation


class TestText(BaseTest):

    # Verify test usage example
    def test_usage_example(self):
        data = pd.DataFrame({"article_1": ["article text here"],
                             "article_2": ["article text here"],
                             "article_3": ["article text here"],
                             "article_4": ["article text here"],
                             "article_5": ["article text here"]})
        labels = pd.DataFrame({"article_1": [1, 1, 0, 1],
                               "article_2": [0, 1, 0, 0],
                               "article_3": [0, 0, 1, 0],
                               "article_4": [0, 0, 1, 1],
                               "article_5": [1, 1, 1, 0]},
                              index=["label_1", "label_2", "label_3", "label_4"])

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=2,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=1)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Verify the selected indices
        self.assertListEqual(list(selected_features.columns), ['article_1', 'article_3'])

    def test_process_category_data(self):
        # Input dataframe with categories and features
        data = pd.DataFrame({
            "article_1": ["article text here", "article text here", "article text here"],
            "article_2": ["article text here", "article text here", "article text here"],
            "article_3": ["article text here", "article text here", "article text here"],
            "category_1": ["sports", "sports", "international"],
            "category_2": ["sports", "entertainment", "international"],
            "category_3": ["international", "entertainment", "international"]
        })

        # Define categories
        categories = ["category_1", "category_2", "category_3"]

        # Call process_category_data function
        matrix = process_category_data(data, categories)

        # Verify matrix and features have the expected shapes and values
        expected_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
        self.assertEqual(matrix.shape, expected_matrix.shape)
        assert np.array_equal(matrix, expected_matrix)

    ################################################
    ########## Tests for random selection ##########
    ################################################

    # Check Random with unicost and diverse (the same features should select)
    # Check solution for large trails with the same seed (the same features should select)
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

        method_unicost = SelectionMethod.TextBased(num_features=3,
                                                   optimization_method="random",
                                                   cost_metric="unicost",
                                                   trials=1)

        method_diverse = SelectionMethod.TextBased(num_features=3,
                                                   optimization_method="random",
                                                   cost_metric="diverse",
                                                   trials=1)

        # Find the best solution using multiple trails and assert that it is selected with the same seed
        method = SelectionMethod.TextBased(num_features=3,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=20)

        selector_unicost = Selective(method_unicost)
        selector_unicost.fit(data, labels)
        selected_features_unicost = selector_unicost.transform(data)

        selector_diverse = Selective(method_diverse)
        selector_diverse.fit(data, labels)
        selected_features_diverse = selector_diverse.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features_unicost.equals(selected_features_diverse))

        # Set the same seed again
        best_selector = Selective(method)
        best_selector.fit(data, labels)
        best_selected_features = best_selector.transform(data)

        # Verify the selected indices
        self.assertListEqual(list(best_selected_features.columns), ['item1', 'item2', 'item3'])

    # Verify selection for the Random method, unicost, and none number of features
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

        method = SelectionMethod.TextBased(num_features=None,
                                           optimization_method="random",
                                           cost_metric="unicost",
                                           trials=1)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Check whether the selector.transform() is returned a DataFrame or not
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Check whether the number of columns in `selected_features' is equal to
        # num_features selected by solving set cover
        self.assertEqual(selected_features.shape[1], 2)

        # Verify the selected feature
        self.assertListEqual(list(selected_features.columns), ['item2', 'item4'])
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

    # Verify selection for the Random method, diverse, and none number of features
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

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="random",
                                           cost_metric="diverse",
                                           trials=1)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        # Check whether the selector.transform() is returned a DataFrame or not
        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Check whether the number of columns in `selected_features' is equal to num_features selected
        # by solving set cover
        self.assertEqual(selected_features.shape[1], 2)

        # Verify the selected indices
        self.assertListEqual(list(selected_features.columns), ['item2', 'item4'])
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

    ################################################
    ########## Tests for greedy selection ##########
    ################################################

    # Check selection for a single labels column of ones (select column with one)
    # Check selection for infeasible instance (empty)
    # Check selection for label matrix with considerable coverage (same features should select with the same seed)
    def test_text_based_greedy_num_feature_one_or_infeasible_or_max(self):

        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["second one in list of more frequent sentences with some repeated words"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["sentence with a lot of repeated common words and more words to increase frequency"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"]})

        # Labels with a single column with all ones
        labels_single_col = pd.DataFrame({"item1": [1, 0, 0, 0, 0], "item2": [1, 0, 0, 0, 0], "item3": [1, 0, 0, 0, 0],
                                          "item4": [1, 0, 0, 0, 0], "item5": [1, 0, 0, 0, 0]})

        method_single_col = SelectionMethod.TextBased(num_features=1,
                                                      optimization_method="greedy",
                                                      cost_metric="unicost")

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels_single_col.shape[1])

        selector = Selective(method_single_col)
        selector.fit(data, labels_single_col)
        selected_features_single_col = selector.transform(data)
        self.assertTrue(isinstance(selected_features_single_col, pd.DataFrame))

        # Verify greedy selects the item1
        self.assertListEqual(list(selected_features_single_col.columns), ['item1'])

        # Labels for infeasible instance
        labels_infeasible = pd.DataFrame({"item1": [0, 0, 0, 0, 0], "item2": [0, 0, 0, 0, 0], "item3": [0, 0, 0, 0, 0],
                                          "item4": [0, 0, 0, 0, 0], "item5": [0, 0, 0, 0, 0]})

        method_infeasible = SelectionMethod.TextBased(num_features=2,
                                                      optimization_method="greedy",
                                                      cost_metric="unicost")

        selector = Selective(method_infeasible)
        selector.fit(data, labels_infeasible)
        selected_features_infeasible = selector.transform(data)
        self.assertTrue(isinstance(selected_features_infeasible, pd.DataFrame))

        # Verify that there is at least one selected column for each label
        feasible = True
        for label in labels_infeasible.columns:
            if label not in selected_features_infeasible.columns:
                feasible = False
                break

        # Check whether feasible is False
        self.assertFalse(feasible)

        # Label columns with considerable coverage
        labels = pd.DataFrame({"item1": [1, 0, 1, 0, 1], "item2": [0, 1, 0, 1, 1], "item3": [1, 0, 1, 0, 1],
                               "item4": [1, 0, 1, 0, 0], "item5": [0, 1, 0, 1, 1]})

        method_max_cols = SelectionMethod.TextBased(num_features=2,
                                                    optimization_method="greedy",
                                                    cost_metric="unicost")

        # Set the same seed again
        selector = Selective(method_max_cols)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 2)

        # Verify selected features with considerable label coverage
        self.assertListEqual(list(selected_features.columns), ['item1', 'item2'])

    # Verify selection for the Greedy method, diverse with highly correlated features
    # (same features should select with the same seed)
    def test_text_based_greedy_num_feature_unicost_diverse(self):
        # Several highly correlated features but grouped based on labels
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

        method = SelectionMethod.TextBased(num_features=3,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="greedy",
                                           cost_metric="diverse")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertEqual(selector.selection_method.trials, 10)
        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 3)

        # Verify that features are selected are low correlated
        self.assertListEqual(list(selected_features.columns), ['item3', 'item5', 'item7'])

    # Check selection for labels with identity matrix (select all columns)
    def test_text_based_greedy_unicost_diverse_identity(self):
        data = pd.DataFrame(
            {"item1": ["this is a sentences with more common words and more words to increase frequency"],
             "item2": ["third sentence with repeated words as item2"],
             "item3": ["a word for more complexity and longer length"],
             "item4": ["two words for more complexity and longer length"],
             "item5": ["more frequent words with a valid vector and more words to increase frequency"],
             "item6": ["another sentence with similar words as item1"],
             "item7": ["third sentence with repeated words as item1"]})

        # Labels with an identity columns
        labels = pd.DataFrame({"item1": [1, 0, 0, 0, 0, 0, 0], "item2": [0, 1, 0, 0, 0, 0, 0],
                               "item3": [0, 0, 1, 0, 0, 0, 0], "item4": [0, 0, 0, 1, 0, 0, 0],
                               "item5": [0, 0, 0, 0, 1, 0, 0], "item6": [0, 0, 0, 0, 0, 1, 0],
                               "item7": [0, 0, 0, 0, 0, 0, 1]})

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        method = SelectionMethod.TextBased(num_features=None,
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="greedy",
                                           cost_metric="diverse")

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Verify all features are selected
        self.assertListEqual(selected_features.columns.tolist(), data.columns.tolist())

    ################################################
    ########## Tests for kmeans selection ##########
    ################################################

    # Verify selection for the KMeans method and unicost cost metric with the same seed
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
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="kmeans",
                                           cost_metric="unicost")  # Default is diverse

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))
        self.assertEqual(selected_features.shape[1], 2)

        # Check the selected features are a subset of the original columns
        self.assertTrue(set(selected_features.columns).issubset(data.columns))

        # Verify that features are selected are low correlated
        self.assertListEqual(list(selected_features.columns), ['item1', 'item5'])

    # Verify selection for the Random method, unicost cost metric, and none number of features with the same seed
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
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="kmeans",
                                           cost_metric="unicost")

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        self.assertTrue(isinstance(selected_features, pd.DataFrame))

        # Verify that the features selected
        self.assertListEqual(list(selected_features.columns), ['item2', 'item4', 'item5', 'item6', 'item7'])

    # Check the test consistency for KMeans with diverse cost and none number of features
    # (same features should select with the same seed)
    # Check the test with same seed and different TextWiser parameters
    # (different features should select with the same seed)
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
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="kmeans",
                                           cost_metric="diverse")  # Default cost metric is diverse

        # Verify that the number of columns is data and labels match
        self.assertEqual(data.shape[1], labels.shape[1])

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        selector2 = Selective(method)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

    ###############################################
    ########## Tests for exact selection ##########
    ###############################################

    # Verify selection for the Exact method, unicost, and none number of features with the same seed
    # (the same features should select)
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
                                           trials=1)

        method2 = SelectionMethod.TextBased(num_features=None,
                                            optimization_method="exact",
                                            cost_metric="unicost",
                                            trials=20)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        selector2 = Selective(method2)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item1', 'item2', 'item3'])

    # Verify selection for the Exact method, unicost, and fixed number of features with the same seed
    # (the same features should select)
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
                                           trials=1)

        method2 = SelectionMethod.TextBased(num_features=2,
                                            optimization_method="exact",
                                            cost_metric="unicost",
                                            trials=20)

        selector = Selective(method)
        selector.fit(data, labels)
        selected_features = selector.transform(data)

        selector2 = Selective(method2)
        selector2.fit(data, labels)
        selected_features2 = selector2.transform(data)

        # Verify the consistency of selected features with the initial run
        self.assertTrue(selected_features.equals(selected_features2))

        # Verify that the features selected
        self.assertListEqual(list(selected_features2.columns), ['item1', 'item3'])

    # Verify selection for the Exact method, diverse, and none number of features with the same seed
    # (the same features should select)
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
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="exact",
                                           cost_metric="diverse",
                                           trials=1)

        method2 = SelectionMethod.TextBased(num_features=None,
                                            featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                           Transformation.NMF(n_components=10,
                                                                                              random_state=123)),
                                            optimization_method="exact",
                                            cost_metric="diverse",
                                            trials=1)

        selector = Selective(method)
        selector.fit(data, labels)

        selector2 = Selective(method2)
        selector2.fit(data, labels)

        # Verify the consistency of selected features with the initial run
        self.assertEqual(selector._imp.content_selector.set_cover_model.objective_value, 1.639294023699553)
        self.assertEqual(selector2._imp.content_selector.set_cover_model.objective_value, 1.639294023699553)

    # Verify selection for the Exact method, diverse, and fixed number of features with the same seed
    # (the same features should select)
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
                                           featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                          Transformation.NMF(n_components=10,
                                                                                             random_state=123)),
                                           optimization_method="exact",
                                           cost_metric="diverse",
                                           trials=1)  # Default cost metric is diverse

        method2 = SelectionMethod.TextBased(num_features=2,
                                            featurization_method=TextWiser(Embedding.TfIdf(min_df=0.),
                                                                           Transformation.NMF(n_components=10,
                                                                                              random_state=123)),
                                            optimization_method="exact",
                                            cost_metric="diverse",
                                            trials=20)  # Default for trials=10

        selector = Selective(method)
        selector.fit(data, labels)

        selector2 = Selective(method2)
        selector2.fit(data, labels)

        # Verify the consistency of selected features with the initial run
        self.assertEqual(selector._imp.content_selector.set_cover_model.objective_value, 1.639294023699553)
        self.assertEqual(selector2._imp.content_selector.set_cover_model.objective_value, 1.639294023699553)

        self.assertEqual(selector._imp.content_selector.max_cover_model.objective_value, 5)
        self.assertEqual(selector2._imp.content_selector.max_cover_model.objective_value, 5)

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

            Selective(method)

    def test_text_invalid_cost_metric(self):

        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               featurization_method=TextWiser(Embedding.TfIdf(),
                                                                              Transformation.NMF()),
                                               optimization_method="random",
                                               cost_metric="invalid")

            Selective(method)

    def test_text_invalid_featurization(self):
        with self.assertRaises(ValueError):
            method = SelectionMethod.TextBased(num_features=3,
                                               optimization_method="exact",
                                               cost_metric="invalid")

            Selective(method)
