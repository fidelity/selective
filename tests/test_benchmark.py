# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from feature.utils import get_data_label
from feature.selector import SelectionMethod, benchmark, calculate_statistics
from tests.test_base import BaseTest


class TestBenchmark(BaseTest):

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

    def test_benchmark_regression(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, output_filename=None)
        _ = calculate_statistics(score_df, selected_df)

        self.assertListAlmostEqual([0.3053471606764446, 0.27265001037792713, 0.4665292949816222, 0.39871798871431496, 0.20859822017269058],
                                   score_df["corr_pearson"].to_list())

        self.assertListAlmostEqual([0.35337823099497323, 0.27106131570570696, 0.3403608920939985, 0.27919214584187463, 0.22975301078604157],
                                   score_df["corr_kendall"].to_list())

        self.assertListAlmostEqual([0.4175152214130028, 0.304906296660894, 0.3951865754854279, 0.31755353822102084, 0.24402514677988724],
                                   score_df["corr_spearman"].to_list())

        self.assertListAlmostEqual([18556.57163129339, 232.84147870961536, 487.75746169668923, 45.10857562258609, 11.63534210812702],
                                   score_df["univ_anova"].to_list())

        self.assertListAlmostEqual([0, 0, 0, 0, 0],
                                   score_df["univ_chi_square"].to_list())

        self.assertListAlmostEqual([0.38748130501016664, 0.030972311154629395, 0.10332044995904166, 0.02446258804706858, 0.07292532752425096],
                                   score_df["univ_mutual_info"].to_list())

        self.assertListAlmostEqual([0.5374323985142709, 0.01587148752174572, 0.21385807822961317, 0.998452656231538, 0.004701537317550907],
                                   score_df["linear"].to_list())

        self.assertListAlmostEqual([0.1455857089432204, 0.0059868642655759976, 0.0, 0.0, 0.0],
                                   score_df["lasso"].to_list())

        self.assertListAlmostEqual([0.5372781001028991, 0.015872990388928143, 0.21356598255185547, 0.9969097836322982, 0.00470112708581501],
                                   score_df["ridge"].to_list())

        self.assertListAlmostEqual([0.568902676572251, 0.08541691893202849, 0.09569412526413482, 0.07764370838746859, 0.172342570844117],
                                   score_df["random_forest"].to_list())

    def test_benchmark_regression_cv(self):
        data, label = get_data_label(fetch_california_housing())
        data = data.drop(columns=["Latitude", "Longitude", "Population"])

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, cv=5, output_filename=None)
        _ = calculate_statistics(score_df, selected_df)

        # Aggregate scores from different cv-folds
        score_df = score_df.groupby(score_df.index).mean()

        self.assertListAlmostEqual(
            [0.3985598297917597, 0.20850825441037615, 0.4666992870763639, 0.2730917394914155, 0.3055611360799319],
            score_df["corr_pearson"].to_list())

        self.assertListAlmostEqual(
            [0.27919138262585763, 0.2297522353771487, 0.34035483445254094, 0.271057965991026, 0.3533773637341713],
            score_df["corr_kendall"].to_list())

        self.assertListAlmostEqual(
            [0.3175505421971197, 0.2440240099984249, 0.3951769790459652, 0.30489911980256346, 0.41751306633445456],
            score_df["corr_spearman"].to_list())

        self.assertListAlmostEqual(
            [37.957763474707654, 10.500397211273347, 394.06992796875835, 186.57009283545057, 14844.857145567506],
            score_df["univ_anova"].to_list())

        self.assertListAlmostEqual([0, 0, 0, 0, 0],
                                   score_df["univ_chi_square"].to_list())

        self.assertListAlmostEqual(
            [0.024989998940959702, 0.07204597410371019, 0.09992776690255863, 0.02930378660964674, 0.3848375372888359],
            score_df["univ_mutual_info"].to_list())

        self.assertListAlmostEqual(
            [1.0003894283465629, 0.0048958465934836595, 0.2142213590932193, 0.01587069982020226, 0.5376133617371683],
            score_df["linear"].to_list())

        self.assertListAlmostEqual(
            [0.0, 0.0, 0.0, 0.005984992397998495, 0.14554803788718568],
            score_df["lasso"].to_list())

        self.assertListAlmostEqual(
            [0.9984400900940591, 0.004895513801696266, 0.21385430705037084, 0.015872543772210295, 0.5374193112415335],
            score_df["ridge"].to_list())

        self.assertListAlmostEqual(
            [0.07628104935957278, 0.1730032839199575, 0.094369811610183, 0.08557048730299173, 0.570775367807295],
            score_df["random_forest"].to_list())

    def test_benchmark_classification(self):
        data, label = get_data_label(load_iris())

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, output_filename=None)
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

    def test_benchmark_classification_cv(self):
        data, label = get_data_label(load_iris())

        # Benchmark
        score_df, selected_df, runtime_df = benchmark(self.selectors, data, label, cv=5, output_filename=None)
        _ = calculate_statistics(score_df, selected_df)

        # Aggregate scores from different cv-folds
        score_df = score_df.groupby(score_df.index).mean()

        self.assertListAlmostEqual([0.8161221983271784, 0.7871883928143776, 0.7020705184086643, 0.4793198034473529],
                                   score_df["corr_pearson"].to_list())

        self.assertListAlmostEqual([0.6780266710547757, 0.6550828618428932, 0.6125815664695313, 0.35594860548691776],
                                   score_df["corr_kendall"].to_list())

        self.assertListAlmostEqual([0.78225620681015, 0.7652859083343029, 0.7201874607448919, 0.44222588698925963],
                                   score_df["corr_spearman"].to_list())

        self.assertListAlmostEqual([946.9891701851375, 781.7441886012473, 95.65931730842011, 39.49994604080157],
                                   score_df["univ_anova"].to_list())

        self.assertListAlmostEqual([92.9884264821005, 53.62326775665224, 8.659084856298207, 2.9711267637858163],
                                   score_df["univ_chi_square"].to_list())

        self.assertListAlmostEqual([0.994113677302704, 0.9907696444894937, 0.4998955427118911, 0.2298786031192229],
                                   score_df["univ_mutual_info"].to_list())

        self.assertListAlmostEqual([0.22327603204146848, 0.03543066514916661, 0.26254667473769594, 0.506591069316828],
                                   score_df["linear"].to_list())

        self.assertListAlmostEqual([0.280393459805252, 0.9489351779830099, 0.6627768115497065, 0.4761878539373159],
                                   score_df["lasso"].to_list())

        self.assertListAlmostEqual([1.1049393460379105e-15, 2.0872192862952944e-15, 6.504056552595708e-16, 4.218847493575594e-16],
                                   score_df["ridge"].to_list())

        self.assertListAlmostEqual([0.4185294825699565, 0.4472560913161835, 0.10091608418224696, 0.03329834193161316],
                                   score_df["random_forest"].to_list())