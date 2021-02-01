# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

"""
:Author: FMR LLC
This module defines the abstract base class for feature selection algorithms.
"""

import abc
from typing import NoReturn, Tuple

import numpy as np
import pandas as pd


class _BaseSelector(metaclass=abc.ABCMeta):
    """Abstract base class for feature selection

    Attributes
    ----------
    seed: int
        The seed for random state.
    """

    @abc.abstractmethod
    def __init__(self, seed: int):
        """Abstract method.
        """
        self.seed: int = seed

        # Number of features
        self.num_features = None

        # Importance/score for each feature
        self.abs_scores = None

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method
        Returns the transformed data.
        """

    def set_num_features(self, data):
        # Int vs. float number of features
        if isinstance(self.num_features, float):
            self.num_features = int(len(data.columns) * self.num_features)

    def get_top_k(self, data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:

        # When num_feature is float, set the size with ratio from data
        self.set_num_features(data)

        # Top feature indexes based on importances
        ind = np.argpartition(scores, -self.num_features)[-self.num_features:]

        # Sort by min index first to keep the order in the original data
        ind_min_first = sorted(ind)

        # Return dataframe with reduced columns
        return data[data.columns[ind_min_first]].copy()


class _BaseSupervisedSelector(_BaseSelector, metaclass=abc.ABCMeta):
    """Abstract base class for feature selection
    """

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, labels: pd.Series) -> NoReturn:
        """Abstract method
        Fits the selector to the given data.
        """

    def fit_transform(self, data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fits the selector to the given and returns the transformed data.
        """
        self.fit(data, labels)
        return self.transform(data)


class _BaseUnsupervisedSelector(_BaseSelector, metaclass=abc.ABCMeta):
    """Abstract base class for feature selection
    """

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame) -> NoReturn:
        """Abstract method
        Fits the selector to the given data.
        """

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits the selector to the given and returns the transformed data.
        """
        self.fit(data)
        return self.transform(data)


class _BaseDispatcher(metaclass=abc.ABCMeta):
    """Abstract class to dispatch
    appropriate machine learning model depending on the task
    """

    @abc.abstractmethod
    def dispatch_model(self, labels: pd.Series, *args) -> NoReturn:
        """
        Implements the dispatcher logic given the labels (defines the task) and arguments
        It initializes underlying machine learning model
        """

    @abc.abstractmethod
    def get_model_args(self, selection_method) -> Tuple:
        """
        Implements the logic of which arguments to send to the dispatcher
        :return: Tuple of dispatch argument
        """
