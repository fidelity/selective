# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import itertools
from typing import NoReturn

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from feature.base import _BaseUnsupervisedSelector


class _Variance(_BaseUnsupervisedSelector):

    def __init__(self, seed: int, threshold: float):
        super().__init__(seed)

        # Track columns that meet variance threshold
        self.keep_features = None
        self.imp: VarianceThreshold = VarianceThreshold(threshold)

    def fit(self, data: pd.DataFrame) -> NoReturn:

        # Fit data
        self.imp.fit(data)

        # Boolean support mask for columns above threshold
        support = self.imp.get_support()

        # Set importance as variances
        self.abs_scores = self.imp.variances_

        # Store Feature names above threshold
        self.keep_features = list(itertools.compress(data.columns, support))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Reduced numpy matrix
        X_reduced = self.imp.transform(data)

        # Return dataframe with reduced column names
        return pd.DataFrame(X_reduced, columns=self.keep_features)
