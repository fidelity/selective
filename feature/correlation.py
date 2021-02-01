# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import numpy as np
import pandas as pd

from feature.base import _BaseUnsupervisedSelector


class _Correlation(_BaseUnsupervisedSelector):

    def __init__(self, seed: int, threshold: float, method: str):
        super().__init__(seed)

        # Track columns that meet correlation threshold
        self.threshold = threshold

        # Correlation method
        self.method = method

        # Create correlation matrix
        self.corr_matrix = None

    def fit(self, data: pd.DataFrame):

        # Find absolute Pearson correlation between pairs of features
        self.corr_matrix = data.corr(method=self.method).abs()

        # Set absolute importance as mean correlation
        # Convert from series to numpy
        self.abs_scores = self.corr_matrix.mean(0).values

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select upper triangle of correlation matrix
        # Diagonals and the rest becomes NaN
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))

        # Find features to drop with any correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]

        # Drop features
        return data.drop(to_drop, axis=1)
