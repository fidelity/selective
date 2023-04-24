# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from feature.utils import get_data_label
from feature.selector import SelectionMethod, benchmark
from tests.test_base import BaseTest


class TestParallel(BaseTest):
    num_features = 3
    corr_threshold = 0.5
    alpha = 1000
    tree_params = {"random_state": 123, "n_estimators": 100}

    selectors = {
        "corr_pearson": SelectionMethod.Correlation(corr_threshold, method="pearson"),
        "corr_kendall": SelectionMethod.Correlation(corr_threshold, method="kendall"),
        "corr_spearman": SelectionMethod.Correlation(corr_threshold, method="spearman"),
        "univ_anova": SelectionMethod.Statistical(num_features, method="anova"),
        "univ_chi_square": SelectionMethod.Statistical(num_features, method="chi_square"),
        "univ_mutual_info": SelectionMethod.Statistical(num_features, method="mutual_info"),
        "linear": SelectionMethod.Linear(num_features, regularization="none"),
        "lasso": SelectionMethod.Linear(num_features, regularization="lasso", alpha=alpha),
        "ridge": SelectionMethod.Linear(num_features, regularization="ridge", alpha=alpha),
        "random_forest": SelectionMethod.TreeBased(num_features),
        "xgboost_clf": SelectionMethod.TreeBased(num_features, estimator=XGBClassifier(**tree_params)),
        "xgboost_reg": SelectionMethod.TreeBased(num_features, estimator=XGBRegressor(**tree_params)),
        "extra_clf": SelectionMethod.TreeBased(num_features, estimator=ExtraTreesClassifier(**tree_params)),
        "extra_reg": SelectionMethod.TreeBased(num_features, estimator=ExtraTreesRegressor(**tree_params)),
        "lgbm_clf": SelectionMethod.TreeBased(num_features, estimator=LGBMClassifier(**tree_params)),
        "lgbm_reg": SelectionMethod.TreeBased(num_features, estimator=LGBMRegressor(**tree_params)),
        "gradient_clf": SelectionMethod.TreeBased(num_features, estimator=GradientBoostingClassifier(**tree_params)),
        "gradient_reg": SelectionMethod.TreeBased(num_features, estimator=GradientBoostingRegressor(**tree_params)),
        "adaboost_clf": SelectionMethod.TreeBased(num_features, estimator=AdaBoostClassifier(**tree_params)),
        "adaboost_reg": SelectionMethod.TreeBased(num_features, estimator=AdaBoostRegressor(**tree_params)),
        "catboost_clf": SelectionMethod.TreeBased(num_features,
                                                  estimator=CatBoostClassifier(**tree_params, silent=True)),
        "catboost_reg": SelectionMethod.TreeBased(num_features, estimator=CatBoostRegressor(**tree_params, silent=True))
    }

    def test_benchmark_regression(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        # Benchmark
        score_df_sequential, selected_df_sequential, runtime_df_sequential = benchmark(self.selectors, data, label)
        score_df_p1, selected_df_p1, runtime_df_p1 = benchmark(self.selectors, data, label, verbose=True, n_jobs=1)
        score_df_p2, selected_df_p2, runtime_df_p2 = benchmark(self.selectors, data, label, verbose=True, n_jobs=2)

        # Scores
        self.assertListAlmostEqual(
            [0.5374323985142709, 0.01587148752174572, 0.21385807822961317, 0.998452656231538, 0.004701537317550907],
            score_df_sequential["linear"].to_list())
        self.assertListAlmostEqual([0.1455857089432204, 0.0059868642655759976, 0.0, 0.0, 0.0],
                                   score_df_sequential["lasso"].to_list())

        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p1["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p2["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p1["lasso"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p2["lasso"].to_list())

        # Selected
        self.assertListEqual([1, 0, 1, 1, 0], selected_df_sequential["linear"].to_list())
        self.assertListEqual([1, 1, 0, 0, 1], selected_df_sequential["lasso"].to_list())

        self.assertListEqual(selected_df_sequential["linear"].to_list(), selected_df_p1["linear"].to_list())
        self.assertListEqual(selected_df_sequential["linear"].to_list(), selected_df_p2["linear"].to_list())
        self.assertListEqual(selected_df_sequential["lasso"].to_list(), selected_df_p1["lasso"].to_list())
        self.assertListEqual(selected_df_sequential["lasso"].to_list(), selected_df_p2["lasso"].to_list())

    def test_benchmark_classification(self):
        data, label = get_data_label(load_iris())

        # Benchmark
        score_df_sequential, selected_df_sequential, runtime_df_sequential = benchmark(self.selectors, data, label)
        score_df_p1, selected_df_p1, runtime_df_p1 = benchmark(self.selectors, data, label, n_jobs=1)
        score_df_p2, selected_df_p2, runtime_df_p2 = benchmark(self.selectors, data, label, n_jobs=2)

        # Scores
        self.assertListAlmostEqual([0.289930, 0.560744, 0.262251, 0.042721],
                                   score_df_sequential["linear"].to_list())
        self.assertListAlmostEqual([0.764816, 0.593482, 0.365352, 1.015095],
                                   score_df_sequential["lasso"].to_list())

        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p1["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p2["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p1["lasso"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p2["lasso"].to_list())

        # Selected
        self.assertListEqual([1, 1, 1, 0], selected_df_sequential["linear"].to_list())
        self.assertListEqual([1, 1, 0, 1], selected_df_sequential["lasso"].to_list())

        self.assertListEqual(selected_df_sequential["linear"].to_list(), selected_df_p1["linear"].to_list())
        self.assertListEqual(selected_df_sequential["linear"].to_list(), selected_df_p2["linear"].to_list())
        self.assertListEqual(selected_df_sequential["lasso"].to_list(), selected_df_p1["lasso"].to_list())
        self.assertListEqual(selected_df_sequential["lasso"].to_list(), selected_df_p2["lasso"].to_list())

    def test_benchmark_regression_cv(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        # Benchmark
        score_df_sequential, selected_df_sequential, runtime_df_sequential = benchmark(self.selectors, data, label,
                                                                                       cv=5, output_filename=None)
        score_df_p1, selected_df_p1, runtime_df_p1 = benchmark(self.selectors, data, label, cv=5,
                                                               output_filename=None, n_jobs=1)
        score_df_p2, selected_df_p2, runtime_df_p2 = benchmark(self.selectors, data, label, cv=5,
                                                               output_filename=None, n_jobs=2)

        # Aggregate scores from different cv-folds
        score_df_sequential = score_df_sequential.groupby(score_df_sequential.index).mean()
        score_df_p1 = score_df_p1.groupby(score_df_p1.index).mean()
        score_df_p2 = score_df_p2.groupby(score_df_p2.index).mean()

        # Scores
        self.assertListAlmostEqual(
            [1.0003894283465629, 0.0048958465934836595, 0.2142213590932193, 0.01587069982020226, 0.5376133617371683],
            score_df_sequential["linear"].to_list())
        self.assertListAlmostEqual([0.0, 0.0, 0.0, 0.005984992397998495, 0.14554803788718568],
                                   score_df_sequential["lasso"].to_list())

        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p1["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p2["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p1["lasso"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p2["lasso"].to_list())

    def test_benchmark_classification_cv(self):
        data, label = get_data_label(load_iris())

        # Benchmark
        score_df_sequential, selected_df_sequential, runtime_df_sequential = benchmark(self.selectors, data, label,
                                                                                       cv=5, output_filename=None)
        score_df_p1, selected_df_p1, runtime_df_p1 = benchmark(self.selectors, data, label, cv=5,
                                                               output_filename=None, n_jobs=1)
        score_df_p2, selected_df_p2, runtime_df_p2 = benchmark(self.selectors, data, label, cv=5,
                                                               output_filename=None, n_jobs=2)

        # Aggregate scores from different cv-folds
        score_df_sequential = score_df_sequential.groupby(score_df_sequential.index).mean()
        score_df_p1 = score_df_p1.groupby(score_df_p1.index).mean()
        score_df_p2 = score_df_p2.groupby(score_df_p2.index).mean()

        # Scores
        self.assertListAlmostEqual([0.223276, 0.035431, 0.262547, 0.506591],
                                   score_df_sequential["linear"].to_list())
        self.assertListAlmostEqual([0.280393, 0.948935, 0.662777, 0.476188],
                                   score_df_sequential["lasso"].to_list())

        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p1["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["linear"].to_list(), score_df_p2["linear"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p1["lasso"].to_list())
        self.assertListAlmostEqual(score_df_sequential["lasso"].to_list(), score_df_p2["lasso"].to_list())
