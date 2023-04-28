# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

"""
:Author: FMR LLC

This module provides a number of constants and helper functions.
"""

from typing import Dict, Union, NamedTuple, NoReturn

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 123456
    """The default random seed."""


def argmax(dictionary: Dict[Num, Num]) -> Num:
    """
    Returns the first key with the maximum value.
    """
    return max(dictionary, key=dictionary.get)


def check_true(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def get_data_label(sklearn_dataset):
    data = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    label = pd.Series(sklearn_dataset.target)
    return data, label


def get_task_string(labels: pd.Series):

    if labels is None:
        return "unsupervised_"

    return "classification_" if is_classification(labels) else "regression_"


def is_classification(labels: pd.Series):
    # all integer labels
    return (labels % 1 == 0).all()


def get_selector(score_func, k: Union[int, float]):

    # Top K or Top Percentile
    if isinstance(k, int):
        return SelectKBest(score_func, k=k)
    elif isinstance(k, float):
        return SelectPercentile(score_func, percentile=int(k * 100))


def normalize_columns(df: pd.DataFrame):
    df_normalized = df / df.sum(axis=0)
    return df_normalized.fillna(0)


class _CapFloor:
    """
    Cap and floor outlier values in contexts based on lower and upper percentiles for each feature.
    """

    def __init__(self, p_low: float = 0.001, p_high: float = 0.999):
        """
        Initialize the given parameters.

        :param p_low: Lower percentile used to compute floor. Float.
        :param p_high: Upper percentile used to compute cap. Float.
        """
        self.p_low = p_low
        self.p_high = p_high

        # Initialize
        self.floor = None
        self.cap = None

    def fit(self, contexts: np.ndarray):
        """
        Compute cap and floor values based on input data.

        :param contexts: Input contexts. Array-like.
        """
        self.floor = np.quantile(contexts, q=self.p_low, axis=0)
        self.cap = np.quantile(contexts, q=self.p_high, axis=0)

    def transform(self, contexts: np.ndarray):
        """
        Cap and floor contexts based on fitted cap and floor values.

        :param contexts: Input contexts. Array-like.
        :return: contexts: Transformed contexts array.
        """
        return np.clip(contexts, self.floor, self.cap)

    def fit_transform(self, contexts: np.ndarray):
        self.fit(contexts)
        return self.transform(contexts)


class DataTransformer:
    """
    Performs standard pre-processing of input contexts used for training and scoring.

    The pipeline includes the following steps:
        - Median imputation of missing values
        - Capping and flooring of outlier values using a percentile-based approach
        - Scaling features to have zero mean and unit variance
    """

    def __init__(self):

        # Imputation
        self.imp = SimpleImputer(strategy='median')

        # Capping and flooring
        self.cp = _CapFloor(p_low=0.005, p_high=0.995)

        # Scaler
        self.scaler = StandardScaler()

        # Initialize fit status
        self.is_fit = False

    def fit(self, contexts: Union[np.ndarray, pd.Series, pd.DataFrame]):
        """
        Fits the individual data transformers using the input contexts.

        :param contexts: Input contexts. Array-like.
        """

        # Impute missing values
        contexts = self.imp.fit_transform(contexts)

        # Cap and floor outlier values
        contexts = self.cp.fit_transform(contexts)

        # Standardize scale of contexts
        self.scaler.fit_transform(contexts)

        # Set fit flag to true
        self.is_fit = True

    def transform(self, contexts: Union[np.ndarray, pd.Series, pd.DataFrame]):
        """
        Transforms the input contexts using the trained transformers.

        :param contexts: Input contexts. Array-like.
        :return: contexts: Transformed contexts. Array-like.
        """

        # Check that transformers have been fit
        if not self.is_fit:
            raise ValueError("Fit must be called before transform.")

        contexts = self.imp.transform(contexts)
        contexts = self.cp.transform(contexts)
        contexts = self.scaler.transform(contexts)

        return contexts

    def fit_transform(self, contexts: Union[np.ndarray, pd.Series, pd.DataFrame]):
        self.fit(contexts)
        return self.transform(contexts)


def reduce_memory(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Convert data types to save memory.
    Return smaller df

    df: is a dataframe whose memory usage is to be reduced

    """

    memory_before = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("memory_before [mb]:", memory_before)

    na_list = []
    for i, col in enumerate(df.columns):
        # Exclude strings
        if df[col].dtype != object:

            # Print current column type
            if verbose:
                print(20*"=")
                print("Column ", i, ":", col)
                print("dtype_before: ", df[col].dtype)

            # make variables for Int, max and min
            is_integer = False
            max_of_col = df[col].max()
            min_of_col = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                na_list.append(col)
                df[col].fillna(min_of_col - 1, inplace=True)

            # Integer conversion test
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_integer = True

            # Make Integer/unsigned Integer datatypes
            if is_integer:
                if min_of_col >= 0:
                    if max_of_col < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif max_of_col < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif max_of_col < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if min_of_col > np.iinfo(np.int8).min and max_of_col < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif min_of_col > np.iinfo(np.int16).min and max_of_col < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif min_of_col > np.iinfo(np.int32).min and max_of_col < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif min_of_col > np.iinfo(np.int64).min and max_of_col < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            if verbose:
                print("dtype_after: ", df[col].dtype)
                print(20*"=")

    memory_after = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print("memory_after [mb]:", memory_after)
        print("memory_reduction [%]:", str(round(100 - 100 * memory_after / memory_before, 2)))

    return df
