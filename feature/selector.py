#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

"""
:Author: FMR LLC
:Version: 1.1.0 of June 16, 2021

This module defines the public interface of the **Selective Library** for feature selection.
"""

import multiprocessing as mp
from time import time
from typing import Dict, Union, NamedTuple, NoReturn, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from feature.base import _BaseDispatcher, _BaseSupervisedSelector, _BaseUnsupervisedSelector
from feature.correlation import _Correlation
from feature.linear import _Linear
from feature.statistical import _Statistical
from feature.tree_based import _TreeBased
from feature.utils import Num, check_true, Constants, normalize_columns
from feature.variance import _Variance

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


__author__ = "FMR LLC"
__version__ = "1.0.0"
__copyright__ = "Copyright (C), FMR LLC"


class SelectionMethod(NamedTuple):

    class Correlation(NamedTuple):
        """
        Unsupervised feature selector that removes high (absolute) correlated
        features using the default Pearson correlation, Kendall Tau or Spearman correlation.

        Both highly positive and highly negative correlations are candidates for removal.
        Given a pair of highly correlated features,
        the first one with smaller index is kept and the other one is removed.

        Pearson correlation coefficients measure the strength of a linear relationship between two variables.
        Values always range between -1 (strong negative relationship) and +1 (strong positive relationship).
        Values at or close to zero imply weak or no linear relationship.
        Correlation coefficient values less than +0.8 or greater than -0.8 are not considered significant.

        Pearson is parametric while Kendall Tau and Spearman are non-parametric ranking methods.

        The strength of the relationship between X and Y is
        sometimes expressed by squaring the correlation coefficient and multiplying by 100.
        The resulting statistic is known as variance explained (or R2).
        Example: a correlation of 0.5 means 0.5^2x100 = 25% of the variance in Y is "explained" or predicted X.

        Randomness:
        Behavior is deterministic, does not depend on seed.

        Attributes
        ----------
        threshold: Num, optional
            Features with higher absolute correlation than this threshold will be removed.
            The default is to keep all features.
        method: str, optional
            Method of correlation:
            * pearson : standard correlation coefficient (default)
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
        """
        threshold: Num = 0.
        method: str = "pearson"

        def _validate(self):
            check_true(isinstance(self.threshold, (int, float)), TypeError("Threshold must a non-negative number."))
            check_true(self.threshold >= 0, ValueError("Threshold must be greater or equal to zero."))
            check_true(self.threshold <= 1, ValueError("Threshold must be less or equal to one."))
            check_true(self.method in ["pearson", "kendall", "spearman"],
                       ValueError("Method of correlation can be pearson, kendall, or spearman."))

    class Linear(NamedTuple):
        """
        Linear Regression for (X, Y)
        Suited for data that is not noisy or
        there is a lot of data compared to the number of features
        and the features are relatively independent.

        Multi-Collinearity Problem:
        When there are multiple (linearly) correlated features linear model becomes unstable.
        In other words, small changes in data can cause large changes
        in the model coefficients making interpretation difficult.

        Regularization:
        Regularization with additional constraints/penalty to prevent overfitting and improve generalization.
        Good for feature selection and interpretation:
            - Lasso produces sparse solutions.
                It is useful for selecting a strong subset of features
            - Ridge regression is good for data interpretation due to its stability.
                Useful features tend to have non-zero coefficients.

        Loss function becomes minimize E(X,Y) + α∥w∥, where
            - w is the vector of model coefficients,
            - ∥⋅∥ is typically L1 or L2 norm, and
            - α is a tunable free parameter, specifying the amount of regularization
            - α = 0 implies an unregularized model.

        L1 Lasso Regularization:
        L1-norm regularization adds a penalty α∑|wi| to the loss function
        Since each non-zero coefficient adds to the penalty, it forces weak features to have zero as coefficients.
        L1 regularization results in sparse solutions
        L1 regularized regression can be unstable on small data changes
        The coefficients and feature ranks can vary significantly between datasets with small deltas
        This is especially the case when there are correlated features.
        L2 regularization deals with this problem.

        L2 Ridge Regularization:
        L2 norm adds the penalty term (α∑wi^2) to the loss function.
        Since the coefficients are squared in the penalty expression, it forces the coefficient to be spread out more equally.
        Correlated features end up receiving similar coefficients, which can lead to stable results.
        Compared to L1 models, coefficients do not differ as much on small delta changes
        In L2 regularization, a predictive feature gets a non-zero coefficient,
        hence it can be used for feature interpretation.

        Notes on Randomness:
            - Linear Regression with no regularization is deterministic. The rest is non-deterministic
            - Linear Regression with lasso is non-deterministic, depends on seed
            - Linear Regression with ridge is non-deterministic, depends on seed
            - Logistic Regression with no regularization is non-deterministic, depends on seed
            - Logistic Regression with lasso is non-deterministic, depends on seed
            - Logistic Regression with ridge is non-deterministic, depends on seed

        Attributes
        ----------
        num_features: Num, optional
            If integer, select top num_features.
            If float, select the top num_features percentile.
        regularization: str, optional
            If lasso, l1-norm regularization is applied.
            If ridge, l2-norm regularization is applied.
            Default is no regularization
        alpha: Num, optional
            Regularization coefficient.
            Default value is one.
        """
        num_features: Num = 0.0
        regularization: str = "none"
        alpha: Num = 1.0

        def _validate(self):
            check_true(isinstance(self.num_features, (int, float)), TypeError("Num features must a number."))
            check_true(self.num_features > 0, ValueError("Num features must be greater than zero."))
            if isinstance(self.num_features, float):
                check_true(self.num_features <= 1, ValueError("Num features ratio must be between [0..1]."))
            check_true(self.regularization in ["none", "lasso", "ridge"],
                       ValueError("Regularization can only be none, lasso, or ridge."))
            check_true(isinstance(self.alpha, (int, float)), TypeError("Alpha must a number."))
            check_true(self.alpha >= 0, ValueError("Alpha cannot be negative"))

    class Statistical(NamedTuple):
        """
        Supervised feature selector based on statistical tests.
        It scores each feature against a given target.

        ANOVA selector can be applied to both regression (F-Test) and classification problems (ANOVA F-Test).
        ChiSquare selector can only be applied to classification problems.
        Mutual Information selector can be applied to both regression and classification problems.

        The F-value ANOVA examines whether the mean of each feature group differ significantly.
        The F-test estimates the degree of linear dependency between two random variables.

        ChiSquare should only be used for non-negative data.
        ChiSquare is useful particularly for NLP methods such as bag-of-words or tf-idf based features.
        This score can be used to select the n_features features with the
        highest values for the test chi-squared statistic from the X matrix.
        X should only contain non-negative features (e.g., term counts in document classification).

        The chi-square test measures dependence between stochastic variables.
        As such it can detect features that are independent of class label.

        Mutual information (MI) between two random variables measures dependency as non-negative value.
        MI equals to zero if and only if two random variables are independent.
        The higher the MI values the higher the dependency.
        Mutual information, as a nonparametric approach, might require more data points for accurate estimation.

        MI might not be best suited for feature ranking as it is not a "metric" and not normalized.
        MI value does not lie within a fixed range.
        As such, MI values cannot be compared between two datasets.
        For continuous variables binning/discretaion is required but
        the results might be sensitive to exact bin selection.

        Maximal information score (MIC) tries to address these gaps by
        searching for the optimal binnign strategy.

        Notes on Randomness:
            - Mutual Info is non-deterministic, depends on the seed value.
            - The other methods are deterministic

        Attributes
        ----------
        num_features: Num, optional
            If integer, select top num_features.
            If float, select the top num_features percentile.
        method: str, optional
            Statistical analysis:
            * anova: Anova and Anova F-test (default)
            * chi_square: Chi-Square
            * mutual_info: Mutual Information score
            * maximal_info: Maximal Information score (MIC)
            * variance_inflation: Variance Inflation factor (VIF)
        """
        num_features: Num = 0.0
        method: str = "anova"

        def _validate(self):
            check_true(isinstance(self.num_features, (int, float)), TypeError("Num features must a number."))
            check_true(self.num_features > 0, ValueError("Num features must be greater than zero."))
            if isinstance(self.num_features, float):
                check_true(self.num_features <= 1, ValueError("Num features ratio must be between [0..1]."))
            check_true(self.method in ["anova", "chi_square", "mutual_info", "maximal_info", "variance_inflation"],
                       ValueError("Statistical method can only be anova, chi_square, mutual_info, or maximal_info."))

    class TreeBased(NamedTuple):
        """
        Tree-based methods for (X, Y) which uses RandomForestRegressor and RandomForestClassifier

        Randomness:
        Behavior is non-deterministic, depends on seed

        Attributes
        ----------
        num_features : Num, optional
            If integer, select top num_features.
            If float, select the top num_features percentile.
        estimator : tree-model, xgboost, ligthgbm, catboost
        """
        num_features: Num = 0.0
        estimator: Optional[Union[RandomForestRegressor, RandomForestClassifier,
                                  XGBClassifier, XGBRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor,
                                  LGBMClassifier, LGBMRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor,
                                  AdaBoostClassifier, AdaBoostRegressor,
                                  CatBoostClassifier, CatBoostRegressor]] = None

        def _validate(self):
            check_true(isinstance(self.num_features, (int, float)), TypeError("Num features must a number."))
            check_true(self.num_features > 0, ValueError("Num features must be greater than zero."))
            if isinstance(self.num_features, float):
                check_true(self.num_features <= 1, ValueError("Num features ratio must be between [0..1]."))
            if self.estimator is not None:
                check_true(isinstance(self.estimator, (RandomForestRegressor, RandomForestClassifier,
                                                       XGBClassifier, XGBRegressor,
                                                       ExtraTreesClassifier, ExtraTreesRegressor,
                                                       LGBMClassifier, LGBMRegressor,
                                                       GradientBoostingClassifier, GradientBoostingRegressor,
                                                       AdaBoostClassifier, AdaBoostRegressor,
                                                       CatBoostClassifier, CatBoostRegressor)),
                           ValueError("Unknown tree-based estimator" + str(self.estimator)))

    class Variance(NamedTuple):
        """
        Unsupervised Feature selector that removes all low-variance features.

        This feature selection algorithm looks only at the features (X),
        not the desired outputs (y), and can thus be used for unsupervised learning.

        In binary features (i.e. Bernoulli random variables), variance is calculated as:
        Var(x)=p(1−p) where p is the proportion of observations of class 1.
        Therefore, by setting p, we can remove features
        where the vast majority of observations are one class.

        Randomness:
        Behavior is deterministic.
        However when results might differ when threshold = 0 vs. threshold != 0

        Attributes
        ----------
        threshold: Num, optional
            Features with variance lower than this threshold will be removed.
            The default is to keep all features with non-zero variance,
            i.e. remove the features that have the same value in all samples.

            Notice: when threshold is 0.0, sklearn specializes the operator
                    hence the variance score between threshold=0.0 can differ

        """
        threshold: Num = 0.0

        def _validate(self):
            check_true(isinstance(self.threshold, (int, float)), TypeError("Threshold must a non-negative number."))
            check_true(self.threshold >= 0, ValueError("Threshold must be non-negative."))


class Selective:
    """**Selective: Feature Selection Library**

    Selective is a feature selection library providing
    supervised and unsupervised feature selection algorithms
    for regression and classification tasks.

    Attributes
    ----------
    selection_method: list
        The feature selection method.
    seed: int
        The seed for random state.
        Randomness depends on the particular selection method.
        Some methods could be deterministic.
        Default value is set to Constants.default_seed.value.

    Example
    -------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from feature.selector import Selective, SelectionMethod
        >>> data = pd.DataFrame(np.array([[1, 1, 1], [2, 2, 0], [3, 3, 1], [4, 4, 0],
        >>>                               [5, 5, 1], [6, 6, 0], [7, 7, 1], [8, 7, 0], [9, 7, 1]]))
        >>> selector = Selective(SelectionMethod.Correlation())
        >>> transformed_data = selector.fit_transform(data)
        >>> transformed_data.shape
        (9, 1)
    """

    def __init__(self, selection_method: Union[SelectionMethod.Correlation,
                                               SelectionMethod.Linear,
                                               SelectionMethod.TreeBased,
                                               SelectionMethod.Statistical,
                                               SelectionMethod.Variance],
                 seed: int = Constants.default_seed):
        """Initializes a feature selector with the given selection method.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        selection_method: SelectionMethod
            The selection method.

        seed: int, optional
            The seed for random state.
            Randomness depends on the particular selection method.
            Some methods could be deterministic.
            Default value is set to Constants.default_seed.value.

        Raises
        ------
        TypeError:  Seed is not an integer.
        ValueError: Invalid seed value.
        ValueError: Invalid selection method.
        """

        # Validate arguments
        Selective._validate_args(seed, selection_method)

        # Save the arguments
        self.seed = seed
        self.selection_method = selection_method

        # Initialize fit to false
        self._is_initial_fit = False

        # Set the selector implementation
        self._imp: Union[None, _BaseUnsupervisedSelector, _BaseSupervisedSelector] = None
        if isinstance(selection_method, SelectionMethod.Correlation):
            self._imp = _Correlation(self.seed, self.selection_method.threshold, self.selection_method.method)
        elif isinstance(selection_method, SelectionMethod.Linear):
            self._imp = _Linear(self.seed, self.selection_method.num_features,
                                self.selection_method.regularization, self.selection_method.alpha)
        elif isinstance(selection_method, SelectionMethod.TreeBased):
            self._imp = _TreeBased(self.seed, self.selection_method.num_features, self.selection_method.estimator)
        elif isinstance(selection_method, SelectionMethod.Statistical):
            self._imp = _Statistical(self.seed, self.selection_method.num_features, self.selection_method.method)
        elif isinstance(selection_method, SelectionMethod.Variance):
            self._imp = _Variance(self.seed, self.selection_method.threshold)
        else:
            raise ValueError("Unknown Selection Method " + str(selection_method))

    def fit(self, data: pd.DataFrame, labels: Optional[pd.Series] = None) -> NoReturn:

        # Validate
        self._validate_fit(data, labels)

        # Initialize underlying machine learning model, if dispatcher used
        if isinstance(self._imp, _BaseDispatcher):
            self._imp.dispatch_model(labels, self._imp.get_model_args(self.selection_method))

        # Fit depending on the task
        if isinstance(self._imp, _BaseSupervisedSelector):
            self._imp.fit(data, labels)
        else:
            self._imp.fit(data)

        # Activate initial fit flag
        self._is_initial_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before transform"))

        # Return transformed data
        return self._imp.transform(data)

    def fit_transform(self, data: pd.DataFrame, labels: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(data, labels)
        return self.transform(data)

    def get_absolute_scores(self) -> np.ndarray:

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before getting importances"))

        return self._imp.abs_scores

    @staticmethod
    def _validate_args(seed, selection_method) -> NoReturn:
        """
        Validates arguments for the constructor.
        """

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer."))
        check_true(seed >= 0, TypeError("The seed must be a non-negative integer."))

        # Selection Method type
        check_true(isinstance(selection_method, (SelectionMethod.Correlation,
                                                 SelectionMethod.Linear,
                                                 SelectionMethod.TreeBased,
                                                 SelectionMethod.Statistical,
                                                 SelectionMethod.Variance)),
                   TypeError("Unknown selection type: " + str(selection_method) + " " + str(type(selection_method))))

        # Selection method value
        selection_method._validate()

    def _validate_fit(self, data, labels):

        # VIF is a Statistical methods, hence BaseSupervised, but does not need labels
        if isinstance(self._imp, _Statistical) and self.selection_method.method == "variance_inflation":
            pass
        else:
            # Supervised implementors, except VIF, require labels
            if isinstance(self._imp, _BaseSupervisedSelector):
                check_true(labels is not None, ValueError("Labels column cannot be none"))
                check_true(isinstance(labels, pd.Series), ValueError("Labels should be a pandas series/column."))

        if not hasattr(self.selection_method, 'num_features'):
            return

        if not isinstance(self.selection_method.num_features, int):
            return

        # Num features when integer, should be less or equal to size of feature columns
        # When float case is validated when selection method is created
        check_true(self.selection_method.num_features <= len(data.columns),
                   ValueError("num_features cannot exceed size of feature columns " +
                              str(self.selection_method.num_features) + " vs. " +
                              str(len(data.columns))))


def benchmark(selectors: Dict[str, Union[SelectionMethod.Correlation,
                                         SelectionMethod.Linear,
                                         SelectionMethod.TreeBased,
                                         SelectionMethod.Statistical,
                                         SelectionMethod.Variance]],
              data: pd.DataFrame,
              labels: Optional[pd.Series] = None,
              cv: Optional[int] = None,
              output_filename: Optional[str] = None,
              drop_zero_variance_features: Optional[bool] = True,
              verbose: bool = False,
              n_jobs: int = 1,
              seed: int = Constants.default_seed) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Benchmark with a given set of feature selectors.
    Return a tuple of data frames with scores, runtime and selected features for each method.

    Parameters
    ----------
    selectors:  Dict[str, Union[SelectionMethod.Correlation,
                                SelectionMethod.Linear,
                                SelectionMethod.TreeBased,
                                SelectionMethod.Statistical,
                                SelectionMethod.Variance]]
        Dictionary of feature selection methods to benchmark.
    data: pd.DataFrame
        Data of shape (n_samples, n_features) used for feature selection.
    labels: pd.Series, optional (default=None)
        The target values (class labels in classification, real numbers in regression).
    cv: int, optional (default=None)
        Number of folds to use for cross-validation.
    output_filename: str, optional (default=None)
        If not None, benchmarking output is saved.
        If file exists, results are appended, otherwise file is created.
    drop_zero_variance_features: bool, optional (default=True)
        Whether to drop features with zero variance before running feature selector methods or not.
    verbose: bool, optional (default=False)
        Whether to print progress messages or not.
    n_jobs: int, optional (default=1)
        Number of concurrent processes/threads to use in parallelized routines.
        If set to -1, all CPUs are used.
        If set to -2, all CPUs but one are used, and so on.
    seed: int, optional (default=Constants.default_seed)
        The random seed to initialize the random number generator.

    Returns
    -------
    Tuple of data frames with scores, selected features and runtime for each method.
    If cv is not None, the data frames will contain the concatenated results from each fold.
    """

    check_true(selectors is not None, ValueError("Benchmark selectors cannot be none."))
    check_true(data is not None, ValueError("Benchmark data cannot be none."))

    if cv is None:
        return _bench(selectors=selectors,
                      data=data,
                      labels=labels,
                      output_filename=output_filename,
                      drop_zero_variance_features=drop_zero_variance_features,
                      verbose=verbose,
                      n_jobs=n_jobs)
    else:

        # Create K-Fold object
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        # Initialize variables
        t0 = time()
        train_labels, test_labels = None, None
        score_df, selected_df, runtime_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Split data into cv-folds and run _bench for each fold
        if verbose:
            print("\n>>> Running")
        for fold, (train_index, _) in enumerate(kf.split(data)):

            if verbose:
                print("\tFold", fold, "...")

            # Split data, labels into folds
            train_data = data.iloc[train_index]
            if labels is not None:
                train_labels = labels.iloc[train_index]

            # Run benchmark
            score_cv_df, selected_cv_df, runtime_cv_df = _bench(selectors=selectors,
                                                                data=train_data,
                                                                labels=train_labels,
                                                                output_filename=output_filename,
                                                                drop_zero_variance_features=drop_zero_variance_features,
                                                                verbose=False,
                                                                n_jobs=n_jobs)

            # Concatenate data frames
            score_df = pd.concat((score_df, score_cv_df))
            selected_df = pd.concat((selected_df, selected_cv_df))
            runtime_df = pd.concat((runtime_df, runtime_cv_df))

        if verbose:
            print(f"<<< Done! Time taken: {(time() - t0) / 60:.2f} minutes")

        return score_df, selected_df, runtime_df


def _bench(selectors: Dict[str, Union[SelectionMethod.Correlation,
                                      SelectionMethod.Linear,
                                      SelectionMethod.TreeBased,
                                      SelectionMethod.Statistical,
                                      SelectionMethod.Variance]],
           data: pd.DataFrame,
           labels: Optional[pd.Series] = None,
           output_filename: Optional[str] = None,
           drop_zero_variance_features: Optional[bool] = True,
           verbose: bool = False,
           n_jobs: int = 1) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Benchmark with a given set of feature selectors.
    Return a tuple of data frames with scores, runtime and selected features for each method.

    Returns
    -------
    Tuple of data frames with scores, selected features and runtime for each method.
    """

    check_true(selectors is not None, ValueError("Benchmark selectors cannot be none."))
    check_true(data is not None, ValueError("Benchmark data cannot be none."))

    # Output file
    if output_filename is not None:
        output_file = open(output_filename, "a")
    else:
        output_file = None

    # Drop features without any variance
    if drop_zero_variance_features:
        selector = Selective(SelectionMethod.Variance())
        data = selector.fit_transform(data, labels)

    method_to_runtime = {}
    score_df = pd.DataFrame(index=data.columns)
    selected_df = pd.DataFrame(index=data.columns)

    # Find the effective number of jobs
    size = len(selectors.items())
    if n_jobs < 0:
        n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
    n_jobs = min(n_jobs, size)

    # Parallel benchmarks for each method
    output_list = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(_parallel_bench)(
            data, labels, method_name, method, verbose)
        for method_name, method in selectors.items())

    # Collect the output from each method
    for output in output_list:
        for method_name, results_dict in output.items():
            score_df[method_name] = results_dict["scores"]
            selected_df[method_name] = results_dict["selected"]
            method_to_runtime[method_name] = results_dict["runtime"]

            if output_filename is not None:
                output_file.write(method_name + " " + str(method_to_runtime[method_name]) + "\n")
                output_file.write(str(results_dict["selected"]) + "\n")
                output_file.write(str(results_dict["scores"]) + "\n")

    # Format
    runtime_df = pd.Series(method_to_runtime).to_frame("runtime").rename_axis("method").reset_index()

    return score_df, selected_df, runtime_df


def _parallel_bench(data: pd.DataFrame,
                    labels: Optional[pd.Series],
                    method_name: str,
                    method: Union[SelectionMethod.Correlation,
                                  SelectionMethod.Linear,
                                  SelectionMethod.TreeBased,
                                  SelectionMethod.Statistical,
                                  SelectionMethod.Variance],
                    verbose: bool) \
                -> Dict[str, Dict[str, Union[pd.DataFrame, list, float]]]:
    """
    Benchmark with a given set of feature selectors.
    Return a dictionary of feature selection method names with their corresponding scores,
    selected features and runtime.

    Returns
    -------
    Dictionary of feature selection method names with their corresponding scores, selected features
    and runtime.
    """

    selector = Selective(method)
    t0 = time()
    if verbose:
        run_str = "\n>>> Running " + method_name
        print(run_str, flush=True)

    try:
        subset = selector.fit_transform(data, labels)
        scores = selector.get_absolute_scores()
        selected = [1 if c in subset.columns else 0 for c in data.columns]
        runtime = round((time() - t0) / 60, 2)
    except Exception as exp:
        print("Exception", exp)
        scores = np.repeat(0, len(data.columns))
        selected = np.repeat(0, len(data.columns))
        runtime = str(round((time() - t0) / 60, 2)) + " (exception)"
    finally:
        if verbose:
            done_str = f"<<< Done! {method_name} Time taken: {(time() - t0) / 60:.2f} minutes"
            print(done_str, flush=True)

    results_dict = {"scores": scores, "selected": selected, "runtime": runtime}

    return {method_name: results_dict}


def calculate_statistics(scores: pd.DataFrame,
                         selected: pd.DataFrame,
                         columns: Optional[list] = None,
                         ignore_constant: Optional[bool] = True) -> pd.DataFrame:
    """
    Calculate statistics for each feature using scores/selections from list of methods.
    Returns data frame with calculated statistics for each feature.

    Parameters
    ----------
    scores:  pd.DataFrame
        Data frame with scores for each feature (index) and selector (columns).
        Each feature could have multiple rows from different cross-validation folds.
    selected: pd.DataFrame
        Data frame with selection flag for each feature (index) and selector (columns).
        Each feature could have multiple rows from different cross-validation folds.
    columns: list (default=None)
        List of methods (columns) to include in statistics.
        If None, all methods (columns) will be used.
    ignore_constant: bool, optional (default=True)
        Whether to ignore methods with the same score for all features.

    Returns
    -------
    Data frame with statistics for each feature
    """

    check_true(isinstance(scores, pd.DataFrame), ValueError("scores must be a data frame."))
    check_true(isinstance(selected, pd.DataFrame), ValueError("selection must be a data frame."))
    check_true(scores.shape == selected.shape, ValueError("Shapes of scores and selected data frames must match."))
    check_true(np.all(scores.index == selected.index),
               ValueError("Index of score and selection data frames must match."))
    check_true(np.all(scores.columns == selected.columns),
               ValueError("Columns of score and selection data frames must match."))

    # Get columns to use
    if columns is None:
        columns = scores.columns

    # Copy data frames
    scores_df = scores[columns].copy()
    selected_df = selected[columns].copy()

    # Group by feature for CV results
    scores_df = scores_df.groupby(scores_df.index).mean()
    selected_df = selected_df.groupby(selected_df.index).mean()

    # Drop methods with constant scores
    if ignore_constant:
        mask = ~np.isclose(np.var(scores_df, axis=0), 0)
        scores_df = scores_df.loc[:, mask]
        selected_df = selected_df.loc[:, mask]

    # Calculate statistics
    stats_df = pd.DataFrame(index=scores_df.index)
    stats_df["score_mean"] = scores_df.mean(axis=1)
    stats_df["score_mean_norm"] = normalize_columns(scores_df).mean(axis=1)
    stats_df["selection_freq"] = selected_df.sum(axis=1)
    stats_df["selection_freq_norm"] = normalize_columns(selected_df).sum(axis=1)

    # Sort
    stats_df.sort_values(by="score_mean_norm", ascending=False, inplace=True)

    return stats_df


def plot_importance(scores: pd.DataFrame,
                    columns: Optional[list] = None,
                    max_num_features: Optional[int] = None,
                    normalize: Optional[str] = None,
                    ignore_constant: Optional[bool] = True,
                    **kwargs):
    """Plot feature selector scores.

    Parameters
    ----------
    scores: pd.DataFrame
        Data frame with scores for each feature (index) and method (columns).
        Each feature could have multiple rows from different cross-validation folds.
    columns: list (default=None)
        List of methods (columns) to include in statistics.
        If None, all methods (columns) will be used.
    max_num_features: int or None, optional (default=None)
        Max number of top features displayed on plot.
        If None all features will be displayed.
    normalize: bool, optional (default=False)
        Whether to normalize scores such that scores sum to 1 for each column.
        This ensures that scores are comparable between different methods.
    ignore_constant: bool, optional (default=True)
        Whether to ignore columns with the same score for all features.
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with feature scores.
    """

    check_true(isinstance(scores, pd.DataFrame), ValueError("Selector scores must be a data frame."))

    # Get columns to use
    if columns is None:
        columns = scores.columns

    # Make copy of data frame
    # Fill nan with zero
    df = scores[columns].copy()
    df.fillna(0, inplace=True)

    # Group by feature for CV results
    df = df.groupby(df.index).mean()

    # Get normalized scores such that scores for each method sums to 1
    if normalize:
        df = normalize_columns(df)

    # Drop methods with constant scores
    if ignore_constant:
        mask = ~np.isclose(np.var(df, axis=0), 0)
        df = df.loc[:, mask]

    # Set max_num_features to total number of features if None
    if max_num_features is None:
        max_num_features = len(df)

    # Calculate the mean score and sort in descending order
    mean_score = np.mean(df, axis=1)
    index = (-mean_score).argsort().values
    df = df.iloc[index, :]

    # Convert data to long format and plot
    df = df.head(max_num_features).reset_index().melt(id_vars="index")
    ax = sns.catplot(x="index", y="value", data=df, kind="bar", color="darkgreen", **kwargs)
    ax.set_xlabels("feature")
    ax.set_ylabels("score")

    return ax
