# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from functools import partial
from typing import NoReturn, Tuple

from minepy import MINE
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from feature.base import _BaseSupervisedSelector, _BaseDispatcher
from feature.utils import get_selector, Num, get_task_string


class _Statistical(_BaseSupervisedSelector, _BaseDispatcher):

    def __init__(self, seed: int, num_features: Num, method: str):
        super().__init__(seed)

        # Number or percentage of features to select
        self.num_features = num_features

        # Statistical method (anova, chi, mutual)
        self.method = method

        # Statistical type (anova_regression vs. anova_classification)
        self.statistical_type = None

        # Implementor top-k or top-percentile
        self.imp = None

        # Implementor factory
        self.factory = {"regression_anova": f_regression,
                        "regression_chi_square": None,
                        "regression_mutual_info": partial(mutual_info_regression, random_state=self.seed),
                        "regression_maximal_info": MINE(),
                        "classification_anova": f_classif,
                        "classification_chi_square": chi2,
                        "classification_mutual_info": partial(mutual_info_classif, random_state=self.seed),
                        "classification_maximal_info": MINE(),
                        "unsupervised_variance_inflation": variance_inflation_factor}

    def get_model_args(self, selection_method) -> Tuple:

        # Pack model argument
        return selection_method.method

    def dispatch_model(self, labels: pd.Series, *args):

        # Unpack model argument
        method = args[0]

        # Get statistical scoring function
        if method == "variance_inflation":
            score_func = self.factory.get("unsupervised_" + method)
        else:
            score_func = self.factory.get(get_task_string(labels) + method)

        # Check scoring compatibility with task
        if score_func is None:
            raise TypeError(method + " cannot be used for task: " + get_task_string(labels))
        elif isinstance(score_func, MINE) or method == "variance_inflation":
            self.imp = score_func
        else:
            # Set sklearn model selector based on scoring function
            self.imp = get_selector(score_func, self.num_features)

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> NoReturn:

        # Calculate absolute scores depending on the method
        if isinstance(self.imp, MINE):
            self.abs_scores = []
            for col in data.columns:
                self.imp.compute_score(data[col], labels)
                score = self.imp.mic()
                self.abs_scores.append(score)
        elif self.method == "variance_inflation":
            # VIF is unsupervised, regression between data and each feature
            self.abs_scores = np.array([variance_inflation_factor(data.values, i) for i in range(data.shape[1])])
        else:
            # sklearn selector model
            self.imp.fit(X=data, y=labels)

            # Set importance as test scores
            self.abs_scores = self.imp.scores_

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select top-k from data based on abs_scores and num_features
        if self.method == "variance_inflation":
            # Smaller is better for VIF, negate the scores
            return self.get_top_k(data, -1*self.abs_scores)
        else:
            return self.get_top_k(data, self.abs_scores)
