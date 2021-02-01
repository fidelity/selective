# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from typing import NoReturn, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from feature.base import _BaseSupervisedSelector, _BaseDispatcher
from feature.utils import Num, get_task_string


class _Linear(_BaseSupervisedSelector, _BaseDispatcher):

    def __init__(self, seed: int, num_features: Num, regularization: str, alpha:Num):
        super().__init__(seed)

        self.num_features = num_features  # this could be int or float
        self.regularization = regularization
        self.alpha = alpha

        # Implementor is decided when data becomes available in fit()
        self.imp = None

        # Implementor factory
        self.factory = {"regression_none": LinearRegression(),
                        "regression_lasso": Lasso(random_state=self.seed),
                        "regression_ridge": Ridge(random_state=self.seed),
                        # "classification_none": LogisticRegression(penalty="none"), # won't converge most times
                        "classification_none": LogisticRegression(random_state=self.seed,
                                                                  multi_class="auto", solver="liblinear"),
                        "classification_lasso": LogisticRegression(random_state=self.seed, penalty='l1',
                                                                   multi_class="auto", solver="liblinear"),
                        "classification_ridge": RidgeClassifier(random_state=self.seed)}

    def get_model_args(self, selection_method) -> Tuple:

        # Pack model argument
        return selection_method.regularization

    def dispatch_model(self, labels: pd.Series, *args):

        # Unpack model argument
        regularization = args[0]

        # Set linear model
        self.imp = self.factory.get(get_task_string(labels) + regularization)

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> NoReturn:

        # Fit linear model
        self.imp.fit(X=data, y=labels)

        # Set importance as absolute coefficients
        # Note: we are not taking p_values into account
        # That could be a problem where a coefficient can be large but with little confidence
        # Also, categorical features can have high coefficients
        # But that does not necessarily mean they are more important
        # See more discussion here:
        # https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py
        self.abs_scores = abs(self.imp.coef_)

        # LogisticRegression/RidgeClassifier returns a coef_ array of (n_classes, n_features)
        # These coefficients map the importance of the feature for a specific class.
        # One approach is to average the importances
        if isinstance(self.imp, LogisticRegression) or isinstance(self.imp, RidgeClassifier):
            self.abs_scores = abs(self.imp.coef_.mean(0))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select top-k from data based on abs_scores and num_features
        return self.get_top_k(data, self.abs_scores)
