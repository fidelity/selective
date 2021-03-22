# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from typing import NoReturn, Tuple

import pandas as pd
from catboost import CatBoost
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from feature.base import _BaseSupervisedSelector, _BaseDispatcher
from feature.utils import Num, get_task_string


class _TreeBased(_BaseSupervisedSelector, _BaseDispatcher):

    def __init__(self, seed: int, num_features: Num, estimator):
        super().__init__(seed)

        self.num_features = num_features    # this could be int or float
        self.estimator = estimator

        # Implementor is decided when data becomes available in fit()
        self.imp = None

        # Implementor factory
        self.factory = {"regression_": RandomForestRegressor(random_state=self.seed,
                                                             n_estimators=50, max_depth=32, n_jobs=3),
                        "classification_": RandomForestClassifier(random_state=self.seed,
                                                                  n_estimators=50, max_depth=32, n_jobs=3)}

    def get_model_args(self, selection_method) -> Tuple:

        # Pack model argument
        return selection_method.estimator

    def dispatch_model(self, labels: pd.Series, *args):

        # Unpack model argument
        estimator = args[0]

        # Classification/Regression task
        task_str = get_task_string(labels)

        # No estimator is given, set tree model
        if estimator is None:
            self.imp = self.factory.get(task_str)
        else:
            # Custom estimator should be compatible with the task
            if "classification_" in task_str:
                if isinstance(self.estimator, CatBoost):
                    if self.estimator._estimator_type != 'classifier':
                        raise TypeError(str(self.estimator) + " cannot be used for task: " + task_str)
                else:
                    if not isinstance(self.estimator, ClassifierMixin):
                        raise TypeError(str(self.estimator) + " cannot be used for task: " + task_str)
            else:
                if isinstance(self.estimator, CatBoost):
                    if self.estimator._estimator_type != 'regressor':
                        raise TypeError(str(self.estimator) + " cannot be used for task: " + task_str)
                else:
                    if not isinstance(self.estimator, RegressorMixin):
                        raise TypeError(str(self.estimator) + " cannot be used for task: " + task_str)

            self.imp = self.estimator

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> NoReturn:

        # Fit tree model
        self.imp.fit(X=data, y=labels)

        # Set importance as feature importances
        self.abs_scores = self.imp.feature_importances_

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select top-k from data based on abs_scores and num_features
        return self.get_top_k(data, self.abs_scores)
