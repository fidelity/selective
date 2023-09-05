# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3


from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

from feature.selector import Selective, SelectionMethod
from feature.utils import get_data_label, Constants
from tests.test_base import BaseTest


class TestTree(BaseTest):

    def test_tree_estimator_lightgbm_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=LGBMRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveBedrms', 'AveOccup'])

    def test_tree_estimator_lightgbm_classif_top_k(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.TreeBased(num_features=2, estimator=XGBClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)

    def test_tree_estimator_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=RandomForestRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveRooms', 'AveOccup'])

    def test_tree_estimator_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2,
                                           estimator=RandomForestClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_tree_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveRooms', 'AveOccup'])

    def test_tree_regress_top_percentile(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=0.6)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveRooms', 'AveOccup'])

    def test_tree_regress_top_k_all(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=5)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], subset.shape[1])
        self.assertListEqual(list(data.columns), list(subset.columns))

    def test_tree_regress_top_percentile_all(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=1.0)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(data.shape[1], subset.shape[1])
        self.assertListEqual(list(data.columns), list(subset.columns))

    def test_tree_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_tree_classif_top_percentile(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=0.5)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_tree_classif_top_percentile_all(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=1.0)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_tree_classif_top_k_all(self):
        data, label = get_data_label(load_iris())
        
        method = SelectionMethod.TreeBased(num_features=4)
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 4)
        self.assertListEqual(list(subset.columns),
                             ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_tree_invalid_num_features(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=100)
        selector = Selective(method)
        with self.assertRaises(ValueError):
            selector.fit(data, label)

    def test_tree_estimator_xgboost_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=XGBRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)

    def test_tree_estimator_xgboost_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2, estimator=XGBClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)

    def test_tree_estimator_extra_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=ExtraTreesRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveRooms', 'AveOccup'])

    def test_tree_estimator_extra_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2, estimator=ExtraTreesClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_tree_estimator_lgbm_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=LGBMRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveBedrms', 'AveOccup'])

    def test_tree_estimator_lgbm_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2, estimator=LGBMClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['sepal width (cm)', 'petal length (cm)'])

    def test_tree_estimator_gradient_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=GradientBoostingRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'HouseAge', 'AveOccup'])

    def test_tree_estimator_gradient_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2, estimator=GradientBoostingClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])

    def test_tree_estimator_adaboost_regress_top_k(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        method = SelectionMethod.TreeBased(num_features=3, estimator=AdaBoostRegressor(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 3)
        self.assertListEqual(list(subset.columns), ['MedInc', 'AveRooms', 'AveOccup'])

    def test_tree_estimator_adaboost_classif_top_k(self):
        data, label = get_data_label(load_iris())

        method = SelectionMethod.TreeBased(num_features=2, estimator=AdaBoostClassifier(random_state=Constants.default_seed))
        selector = Selective(method)
        selector.fit(data, label)
        subset = selector.transform(data)

        # Reduced columns
        self.assertEqual(subset.shape[1], 2)
        self.assertListEqual(list(subset.columns), ['petal length (cm)', 'petal width (cm)'])
