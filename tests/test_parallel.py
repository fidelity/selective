# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from feature.utils import get_data_label
from feature.selector import SelectionMethod, benchmark, calculate_statistics
from tests.test_base import BaseTest


class TestParallel(BaseTest):

    num_features = 3
    corr_threshold = 0.5
    alpha = 1000
    tree_params = {"random_state": 123, "n_estimators": 100}

    selectors = {
        # "corr_pearson": SelectionMethod.Correlation(corr_threshold, method="pearson"),
        # "corr_kendall": SelectionMethod.Correlation(corr_threshold, method="kendall"),
        # "corr_spearman": SelectionMethod.Correlation(corr_threshold, method="spearman"),
        # "univ_anova": SelectionMethod.Statistical(num_features, method="anova"),
        # "univ_chi_square": SelectionMethod.Statistical(num_features, method="chi_square"),
        # "univ_mutual_info": SelectionMethod.Statistical(num_features, method="mutual_info"),
        "linear": SelectionMethod.Linear(num_features, regularization="none"),
        "lasso": SelectionMethod.Linear(num_features, regularization="lasso", alpha=alpha),
        # "ridge": SelectionMethod.Linear(num_features, regularization="ridge", alpha=alpha),
        # "random_forest": SelectionMethod.TreeBased(num_features),
        # "xgboost_clf": SelectionMethod.TreeBased(num_features, estimator=XGBClassifier(**tree_params)),
        # "xgboost_reg": SelectionMethod.TreeBased(num_features, estimator=XGBRegressor(**tree_params)),
        # "extra_clf": SelectionMethod.TreeBased(num_features, estimator=ExtraTreesClassifier(**tree_params)),
        # "extra_reg": SelectionMethod.TreeBased(num_features, estimator=ExtraTreesRegressor(**tree_params)),
        # "lgbm_clf": SelectionMethod.TreeBased(num_features, estimator=LGBMClassifier(**tree_params)),
        # "lgbm_reg": SelectionMethod.TreeBased(num_features, estimator=LGBMRegressor(**tree_params)),
        # "gradient_clf": SelectionMethod.TreeBased(num_features, estimator=GradientBoostingClassifier(**tree_params)),
        # "gradient_reg": SelectionMethod.TreeBased(num_features, estimator=GradientBoostingRegressor(**tree_params)),
        # "adaboost_clf": SelectionMethod.TreeBased(num_features, estimator=AdaBoostClassifier(**tree_params)),
        # "adaboost_reg": SelectionMethod.TreeBased(num_features, estimator=AdaBoostRegressor(**tree_params)),
        # "catboost_clf": SelectionMethod.TreeBased(num_features, estimator=CatBoostClassifier(**tree_params, silent=True)),
        # "catboost_reg": SelectionMethod.TreeBased(num_features, estimator=CatBoostRegressor(**tree_params, silent=True))
    }

    def test_benchmark_regression(self):
        data, label = get_data_label(load_boston())
        data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, verbose=True, n_jobs=2)
        stats_df = calculate_statistics(score_df, selected_df)

        print(score_df)
        print(selected_df)
        print(runtime_df)
        print(stats_df)

    # def test_benchmark_regression_cv(self):
    #     data, label = get_data_label(load_boston())
    #     data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])
    #
    #     # Benchmark
    #     score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, cv=5, output_filename=None)
    #     _ = calculate_statistics(score_df, selected_df)
    #
    #     # Aggregate scores from different cv-folds
    #     score_df = score_df.groupby(score_df.index).mean()
    #
    # def test_benchmark_classification(self):
    #     data, label = get_data_label(load_iris())
    #
    #     # Benchmark
    #     score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, output_filename=None)
    #     _ = calculate_statistics(score_df, selected_df)
    #
    #
    # def test_benchmark_classification_cv(self):
    #     data, label = get_data_label(load_iris())
    #
    #     # Benchmark
    #     score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, cv=5, output_filename=None)
    #     _ = calculate_statistics(score_df, selected_df)
    #
    #     # Aggregate scores from different cv-folds
    #     score_df = score_df.groupby(score_df.index).mean()
