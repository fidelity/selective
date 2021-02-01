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


class TestBenchmark(BaseTest):

    def test_benchmark_regression(self):

        data, label = get_data_label(load_boston())
        data = data.drop(columns=["CHAS", "NOX", "RM", "DIS", "RAD", "TAX", "PTRATIO", "INDUS"])

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
            "catboost_clf": SelectionMethod.TreeBased(num_features, estimator=CatBoostClassifier(**tree_params, silent=True)),
            "catboost_reg": SelectionMethod.TreeBased(num_features, estimator=CatBoostRegressor(**tree_params, silent=True))
        }

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(selectors, data, label, output_filename=None)
        _ = calculate_statistics(score_df, selected_df)

        self.assertListAlmostEqual([0.4787777784012165, 0.47170429073431874, 0.5596288196730658, 0.4400410275414326, 0.5674082968785575],
                                   score_df["corr_pearson"].to_list())

        self.assertListAlmostEqual([0.5357134888110283, 0.48128808343101986, 0.5132201793752295, 0.3384081264406572, 0.49448886053070107],
                                   score_df["corr_kendall"].to_list())

        self.assertListAlmostEqual([0.6542231557010167, 0.5538583519391704, 0.6267310661636885, 0.3924548536221991, 0.5984933578623318],
                                   score_df["corr_spearman"].to_list())

        self.assertListAlmostEqual([89.48611475768125, 75.25764229895405, 83.47745921923685, 63.05422911249312, 601.6178711099022],
                                   score_df["univ_anova"].to_list())

        self.assertListAlmostEqual([0, 0, 0, 0, 0],
                                   score_df["univ_chi_square"].to_list())

        self.assertListAlmostEqual([0.3421450205863028, 0.1806168920395521, 0.31266011627421086, 0.16107911083428794, 0.666208499757925],
                                   score_df["univ_mutual_info"].to_list())

        self.assertListAlmostEqual([0.06901111285092865, 0.05408618283036938, 0.06145227292569164, 0.006510036424819454, 0.9546615660373198],
                                   score_df["linear"].to_list())

        self.assertListAlmostEqual([0.05682706487290267, 0.051008405488957305, 0.05319245109490162, 0.007176306398647428, 0.9231211889322195],
                                   score_df["lasso"].to_list())

        self.assertListAlmostEqual([0.0690214777400926, 0.054087779998048285, 0.06144441861097637, 0.006510854482697315, 0.95459417786841],
                                   score_df["ridge"].to_list())

        self.assertListAlmostEqual([0.10947144861974874, 0.020211076089938374, 0.08416074180466389, 0.045604950489313435, 0.7405517829963355],
                                   score_df["random_forest"].to_list())

    def test_benchmark_classification(self):

        data, label = get_data_label(load_iris())

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
            "catboost_clf": SelectionMethod.TreeBased(num_features, estimator=CatBoostClassifier(**tree_params, silent=True)),
            "catboost_reg": SelectionMethod.TreeBased(num_features, estimator=CatBoostRegressor(**tree_params, silent=True))
        }

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(selectors, data, label, output_filename=None)
        _ = calculate_statistics(score_df, selected_df)

        self.assertListAlmostEqual([0.7018161715727902, 0.47803395524999537, 0.8157648279049796, 0.7867331225527027],
                                   score_df["corr_pearson"].to_list())

        self.assertListAlmostEqual([0.6127053183332257, 0.35502921869499415, 0.6778502590804124, 0.6548312268837866],
                                   score_df["corr_kendall"].to_list())

        self.assertListAlmostEqual([0.7207411401565564, 0.4413611232398492, 0.7823000090067262, 0.7652468370362326],
                                   score_df["corr_spearman"].to_list())

        self.assertListAlmostEqual([119.26450218449871, 49.16004008961098, 1180.1611822529776, 960.0071468018025],
                                   score_df["univ_anova"].to_list())

        self.assertListAlmostEqual([10.81782087849401, 3.7107283035324987, 116.31261309207022, 67.04836020011116],
                                   score_df["univ_chi_square"].to_list())

        self.assertListAlmostEqual([0.4742659474041446, 0.2458627871667194, 0.9899864089960027, 0.9892550496360593],
                                   score_df["univ_mutual_info"].to_list())

        self.assertListAlmostEqual([0.28992981466266715, 0.5607438535573831, 0.2622507287680856, 0.04272068858604694],
                                   score_df["linear"].to_list())

        self.assertListAlmostEqual([0.7644807315853743, 0.594582626209646, 0.3661598482641388, 1.0152555188158772],
                                   score_df["lasso"].to_list())

        self.assertListAlmostEqual([1.646830819860649e-15, 1.572815951552305e-15, 3.2612801348363973e-15, 5.773159728050814e-15],
                                   score_df["ridge"].to_list())

        self.assertListAlmostEqual([0.09210348279677849, 0.03045933928742506, 0.4257647994615192, 0.45167237845427727],
                                   score_df["random_forest"].to_list())
