# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from typing import NoReturn, Union, List, Tuple
import random
import pandas as pd
import numpy as np
from feature.base import _BaseSupervisedSelector
from feature.utils import Num, Constants
from mip import Model, xsum, minimize, maximize, BINARY, INTEGER, OptimizationStatus
from sklearn.cluster import KMeans
from scipy import sparse
from textwiser import TextWiser


class Data:
    def __init__(self, cost: Union[List, np.ndarray], matrix: Union[List[List], np.ndarray]):
        self._verify_input(cost, matrix)

        # Cost Vector c[variable]
        self.cost = cost

        # Coefficient Matrix M[row][column]
        self.matrix = matrix

        # Range of variables
        self.X = range(len(self.cost))

        # Range of rows
        self.rows = range(self.matrix.shape[0])

        # Range of columns
        self.columns = range(self.matrix.shape[1])

    @staticmethod
    def _verify_input(cost, matrix):
        if cost is None:
            raise ValueError("Cost cannot be None.")
        if matrix is None:
            raise ValueError("Matrix cannot be None.")
        if matrix.ndim != 2:
            raise ValueError("Matrix should be 2D")
        if len(cost) != matrix.shape[1]:
            raise ValueError("Cost vector size should match the number of columns in the matrix: "
                             f"{len(cost)} vs. {matrix.shape[1]}.")


class ContentSelector:
    """ Content Selector.

    This module contains classes for feature selection, in which
    a subset of features is selected from a larger set of features
    in order to maximize the performance of a given model or analysis.

    The module includes four selection algorithms:
        Random selection (random),
        Greedy forward selection (greedy),
        KMeans clustering-based selection (KMeans), and
        Multi-level set-covering optimization-based approach (exact).

    All algorithms can be used with either a uniform cost metric (unicost) or
    a diversity-based cost metric (diversity).

    The general optimization process involves selecting a subset of features
    that maximizes diversity in the text embedding spaces and label coverage
    while minimizing the number of features selected.

    Attributes
    ----------
    selection_size: int
        Number of content/feature to select.
    seed: int
        Random seed.
    trials: int
        Number of trials to run for random functions to estimate average coverage.
    verbose: bool
        Print intermediate results if True.
    matrix: np.ndarray
        Coverage matrix where each row corresponds to all predefined labels for each category and the columns
        correspond to the content.
    features: np.ndarray
        Numeric featurization of content.

    References:
    [1] Kadioglu et. al., Optimized Item Selection to Boost Exploration for Recommender Systems, CPAIOR'21
    [2] Kleynhans et. al., Active Learning Meets Optimized Item Selection, DSO@IJCAI'21

    """

    def __init__(self, selection_size: int, seed: int = Constants.default_seed, trials: int = 10,
                 verbose: bool = False):
        """ContentSelector

       Creates a content selector object.

       Parameters
       ----------
        selection_size: int
            Maximum number of content to select.
        seed: int, default=123456
            Random seed.
        trials: int, default=10
            Number of trials to run for random functions to estimate average coverage.
        verbose: bool, default=False
            Print intermediate results if True.
        """

        # Validate arguments
        self.featurization_method = None
        self.cost_metric = None
        self._validate_args(selection_size)

        # Set input arguments
        self.selection_size = selection_size
        self.seed = seed
        self.trials = trials
        self.verbose = False

        # Initialize class variables
        self.matrix = None
        self.features = None
        self._num_rows = None
        self._num_cols = None

    def run(self, input_df: pd.DataFrame, categories: pd.DataFrame, selection_size: int,
            featurization_method: TextWiser, optimization_method: str = "exact",
            cost_metric: str = "diverse", trials: int = 10) -> List:

        """Run content selection algorithm.

        Parameters
        ----------
        input_df: pd.DataFrame
            Input data frame with categories and features of content to select from.

        categories: pd.DataFrame
            Data frame with data that contains categories/labels to be covered.

        selection_size: int
            Number of feature to select.

        featurization_method: TextWiser

        optimization_method: str, default="exact"
            Optimization method used to perform content selection.
                - See TextBased in Selector.py for supported options definition

        cost_metric: str, default = "diverse"

        trials: int, default = 10

        Returns
        -------
        List of indices in data that are selected.
        """

        # Initialize arguments
        self.selection_size = selection_size
        self.featurization_method = featurization_method
        self.cost_metric = cost_metric

        # Process data
        self._process_data(input_df, categories, selection_size, optimization_method, cost_metric)

        if optimization_method == "exact":
            selected = self._select_multi_level_optimization()
        elif optimization_method == "greedy":
            selected = self._select_greedy()
        elif optimization_method == "kmeans":
            selected = self._select_kmeans()
        elif optimization_method == "random":
            selected = self._select_random(trials=trials)
        else:
            raise NotImplementedError(f"{optimization_method} has not been implemented.")

        # Sort selection
        selected.sort()

        return selected

    def _process_data(self, input_df: pd.DataFrame, categories: pd.DataFrame, selection_size: int,
                      optimization_method: str, cost_metric: str) -> NoReturn:
        """_process_data
        We have implemented two versions of this function. In the version here, the categories parameter
        actually refers to the labels (rather than categories) to be covered. It generates features for input data
        using a specified featurization method, and creates a matrix where each row corresponds to a label and each
        column corresponds to a feature.
        The version of code for generating labels from categories has been moved to utils.py.

        Parameters
        ----------
        input_df: pd.DataFrame
            Input data frame with categories and features of content to select from.

        categories: pd.DataFrame
            Data frame that contains categories/labels to be covered.

        selection_size: int
            Number of feature to select.

        optimization_method: str
            Code prevents performing featurization for the case mentioned in 'option_map' dictionary with
            random, greedy and exact optimization.

        cost_metric: str
            Code prevents performing featurization for the case mentioned in 'option_map' dictionary with
            unicost or diverse cost metric.

        Returns
        -------
        No return
       --------
        """
        self.matrix = categories.to_numpy()
        self._num_rows, self._num_cols = self.matrix.shape

        # option_map contains the tuple of optimization_method, selection_size, and cost_metric
        # that do not need featurization method
        option_map = {
            ("random", True, "diverse"): lambda: None,
            ("random", False, "unicost"): lambda: None,
            ("random", True, "unicost"): lambda: None,
            ("greedy", False, "unicost"): lambda: None,
            ("greedy", True, "unicost"): lambda: None,
            ("exact", True, "unicost"): lambda: None,
            ("exact", False, "unicost"): lambda: None
        }

        # If the entry of in the dictionary matches below argument set, then corresponding lambda function is retrieved
        options = option_map.get((optimization_method, selection_size is not None, cost_metric))
        if options:
            options()
        else:
            feature_column = self.featurization_method.fit_transform(input_df)
            self.features = np.array([eval(txt_feature) if isinstance(txt_feature, str)
                                      else txt_feature for txt_feature in feature_column.tolist()])
            if len(self.features) != self._num_cols:
                raise ValueError(f"Process Data Error: features size ({len(self.features)}) should match the number"
                                 f"of columns ({self._num_cols})")
        if self.matrix.ndim != 2:
            raise ValueError("Process Data Error: matrix should 2D")
        if self.selection_size is not None and self.selection_size > self._num_cols:
            raise ValueError("Process Data Error: selection_size cannot exceed num columns")

    def _select_multi_level_optimization(self) -> List:

        unicost = np.ones(self._num_cols)
        data = Data(cost=unicost, matrix=self.matrix)

        if self.selection_size is None:
            if self.cost_metric == "unicost":
                selected = self._solve_set_cover(data)
            else:
                unicost_selected = self._solve_set_cover(data)
                k_unicost = len(unicost_selected)
                cost_diversity = self._get_diversity_cost(k_unicost)
                data_diversity = Data(cost=cost_diversity, matrix=self.matrix)
                selected = self._solve_set_cover(data_diversity)
        else:
            if self.cost_metric == "unicost":
                selected = self._solve_set_cover(data)
                cost = unicost
                k = len(selected)
            else:
                unicost_selected = self._solve_set_cover(data)
                k_unicost = len(unicost_selected)
                cost_diversity = self._get_diversity_cost(k_unicost)
                data_diversity = Data(cost=cost_diversity, matrix=self.matrix)
                selected = self._solve_set_cover(data_diversity)
                cost = cost_diversity
                k = len(selected)

            if self.selection_size < k:
                data = Data(cost=cost, matrix=self.matrix)
                selected = self._solve_max_cover(data, selected)
            else:
                pass

        # Return solution
        return selected

    def _get_diversity_cost(self, selected_size: int) -> List[float]:
        """
        diversity cost:
            there is a situation where the minimum distance between each two features is zero. This situation may occur
             if the features are very similar or if there are a limited number of features.
              When this happens, the cost becomes zero. There are two ways to handle this situation:
                - add dummy cost to diversity cost (DUMMY)
                - check if any of the distances in the distances matrix is zero, and if so, replace it with
                 the next smallest non-zero distance (NEXT_S_DIS).
        """
        kmeans = KMeans(n_clusters=selected_size, random_state=self.seed, n_init=self.trials)
        distances = kmeans.fit_transform(self.features)

        ### NEXT_S_DIS
        for i in range(self._num_cols):
            if np.min(distances[i]) == 0:
                nonzero_distances = distances[i][np.nonzero(distances[i])]
                if len(nonzero_distances) > 0:
                    min_nonzero_distance = np.min(nonzero_distances)
                    distances[i][distances[i] == 0] = min_nonzero_distance
        ###

        diversity_cost = [np.min(distances[i]) for i in range(self._num_cols)]

        ### DUMMY
        # diversity_cost = [c + 1e-1 for c in diversity_cost]
        ###

        # Scale contexts so that sum of costs remain constant
        diversity_cost = [c * self._num_cols / sum(diversity_cost) for c in diversity_cost]

        return diversity_cost

    def _get_num_row_covered(self, selected) -> int:
        if self.matrix is None:
            raise ValueError("matrix is not initialized")

        # Create indicator to count number of covered rows
        row_covered = np.sum(self.matrix[:, selected], axis=1)

        row_covered[row_covered > 0] = 1

        num_row_covered = sum(row_covered)

        return num_row_covered

    def _get_selection_size(self) -> Tuple[np.ndarray, int]:

        cost = np.ones(self._num_cols) if self.cost_metric == "unicost" else self._get_diversity_cost(self._num_cols)
        data = Data(cost=cost, matrix=self.matrix)
        selection = self._solve_set_cover(data)
        selection_size = len(selection)

        return cost, selection_size

    def _get_closest_to_centroids(self, kmeans: KMeans) -> List:
        df = pd.DataFrame({"cluster": kmeans.labels_})

        for c in np.unique(kmeans.labels_):
            # Get indices of cluster
            mask = df["cluster"] == c

            # Squared error to cluster centroid
            dist = np.sum((self.features[mask] - kmeans.cluster_centers_[c]) ** 2, axis=1)

            # Create column in df
            df.loc[mask, "dist_to_centroid"] = dist

        kmeans_selected_df = df.loc[df.groupby('cluster', sort=False).dist_to_centroid.idxmin()]
        kmeans_selected = kmeans_selected_df.index.values.tolist()

        return kmeans_selected

    def _select_kmeans(self) -> List:
        if self.selection_size is None:
            _, selected_size = self._get_selection_size()
        else:
            selected_size = self.selection_size

        kmeans = KMeans(n_clusters=selected_size, random_state=self.seed, n_init=self.trials)
        kmeans.fit(self.features)
        selected = self._get_closest_to_centroids(kmeans)
        num_row_covered = self._get_num_row_covered(selected)

        if self.verbose:
            print("\nKMEANS SELECTION:", selected_size, " columns to cover rows", self._num_rows)
            print("=" * 40)
            print("SIZE:", len(selected), "reduction: {:.2f}".format((self._num_cols - len(selected)) / self._num_cols))
            print("SELECTED:", selected)
            print("NUM ROWS COVERED:", num_row_covered, "coverage: {:.2f}".format(num_row_covered / self._num_rows))
            print("STATUS: KMEANS")
            print("Cost Metric:", self.cost_metric)
            print("=" * 40)

        return selected

    def _select_greedy(self) -> List:
        if self.selection_size is None:
            cost, selection_size = self._get_selection_size()
        else:
            if self.cost_metric == "unicost":
                cost = np.ones(self._num_cols)
                selection_size = self.selection_size
            else:
                diversity_cost = self._get_diversity_cost(self._num_cols)
                selection_size = self.selection_size
                cost = diversity_cost

        # Compressed sparse column (transposed for convenience)
        sparse_col = sparse.csr_matrix(self.matrix.T, copy=True)

        # Initial guess of the Lagrangian multiplier with greedy algorithm
        adjusted_cost = cost / sparse_col.dot(np.ones(self._num_rows))
        cost_matrix = adjusted_cost * self.matrix + np.amax(adjusted_cost) * (~self.matrix)
        u = adjusted_cost[np.argmin(cost_matrix, axis=1)]

        # Nothing is selected, everything is uncovered
        selected = np.zeros(self._num_cols, dtype=bool)
        iuncovered = np.ones(self._num_rows, dtype=bool)
        score = np.zeros(self._num_cols)

        epsilon = 1E-5
        size = sum(selected)

        # While there are uncovered rows and below selection size
        while np.count_nonzero(iuncovered) > 0 and size < selection_size:

            # Faster than indexing, made possible by sparse_col.dot
            mu = sparse_col.dot(iuncovered.astype(int)).astype(float)
            mu[mu <= epsilon] = epsilon
            # Set Lagrange multiplier zero for covered rows
            u[~iuncovered] = 0
            gamma = (cost - sparse_col.dot(u))
            select_gamma = (gamma >= 0)

            if np.count_nonzero(select_gamma) > 0:
                score[select_gamma] = gamma[select_gamma] / mu[select_gamma]

            if np.count_nonzero(~select_gamma) > 0:
                score[~select_gamma] = gamma[~select_gamma] * mu[~select_gamma]

            # Add new column (column with minimum cost that has not been selected yet
            inewcolumn = (np.nonzero(~selected)[0])[np.argmin(score[~selected])]
            selected[inewcolumn] = True
            size += 1

            iuncovered = ~np.logical_or(~iuncovered, self.matrix[:, inewcolumn])

        if size == selection_size:
            print("Warning: max greedy reached selection size", selection_size)

        # Solution
        selected = list(selected.nonzero()[0])
        num_row_covered = self._get_num_row_covered(selected)

        if self.verbose:
            print("\nGREEDY SELECTION:", len(selected), "columns to cover rows ", self._num_rows)
            print("=" * 40)
            print("SIZE:", len(selected), "reduction: {:.2f}".format((self._num_cols - len(selected)) / self._num_cols))
            print("SELECTED:", selected)
            print("NUM ROWS COVERED:", num_row_covered, "coverage: {:.2f}".format(num_row_covered / self._num_rows))
            print("STATUS: GREEDY")
            print("COST METRIC:", self.cost_metric)
            print("=" * 40)

        return selected

    def _select_random(self, trials: int = 10) -> List[int]:
        # Set the seed
        random.seed(self.seed)
        best_selected = []
        best_covered = 0

        if self.selection_size is None:
            _, self.selection_size = self._get_selection_size()

        for t in range(trials):
            # Select a sample of selection_size without repetition
            selected = []
            while len(selected) < self.selection_size and self._num_cols is not None:
                i = random.randint(0, self._num_cols - 1)
                while i in selected:
                    i = random.randint(0, self._num_cols -1)
                selected.append(i)

            # Count covered categories
            num_row_covered = self._get_num_row_covered(selected)
            if num_row_covered > best_covered:
                best_covered = num_row_covered
                best_selected = selected

        # Calculate coverage metrics
        num_row_covered = self._get_num_row_covered(best_selected)
        coverage = num_row_covered / self._num_rows

        if self.verbose:
            print("\nRANDOM SELECTION:", self.selection_size, "columns to cover rows ", self._num_rows)
            print("=" * 40)
            print("SIZE:", len(best_selected),
                  "reduction: {:.2f}".format((self._num_cols - len(best_selected)) / self._num_cols))
            print("SELECTED:", best_selected)
            print("NUM (AVG) ROWS COVERED:",
                  num_row_covered, "coverage: {:.2f}".format(coverage))
            print("STATUS: RANDOM")
            print("Cost Metric:", self.cost_metric)
            print("=" * 40)
        return best_selected

    def _solve_set_cover(self, data: Data) -> List:

        # Create Model object
        model = Model("Set Cover Model")

        # Variables
        x = [model.add_var(var_type=BINARY) for _ in data.X]

        # Constraint: every row should be covered
        for row in data.rows:
            model.add_constr(xsum(data.matrix[row, i] * x[i] for i in data.X) >= 1)

        # Objective: minimize
        model.objective = minimize(xsum(data.cost[i] * x[i] for i in data.X))

        # Solve
        model.verbose = False
        model.optimize()
        if model.status != OptimizationStatus.OPTIMAL:
            raise ValueError("Max Cover Error: optimal solution not found.")

        # Solution
        selected = [i for i in data.X if float(x[i].x) >= 0.99]

        if self.verbose:
            print("=" * 40)
            print("SET COVER OBJECTIVE:", model.objective_value)
            print("SELECTED:", selected)
            print("STATUS:", model.status)
            print("=" * 40)

        # Return
        return selected

    def _solve_max_cover(self, data: Data, selected: List) -> List:

        # Model
        model = Model("Max Cover Model")

        # Variables
        x = [model.add_var(var_type=BINARY) for _ in data.X]
        is_row_covered = [model.add_var(var_type=BINARY) for _ in data.rows]
        num_row_covered = model.add_var(var_type=INTEGER)

        # Constraint: Link between x and is_row_covered
        for row in data.rows:
            for i in data.X:
                # if any selected column has the label, then the row would be covered
                model.add_constr(data.matrix[row, i] * x[i] <= is_row_covered[row])
            # total selected
            model.add_constr(xsum(data.matrix[row, i] * x[i] for i in data.X) >= is_row_covered[row])

        # Constraint: Link is_row_covered with num_row_covered
        model.add_constr(xsum(is_row_covered[row] for row in data.rows) == num_row_covered)

        # Constraint: If selected is given, discard columns that are not part of selection
        for i in data.X:
            if i not in selected:
                model.add_constr(x[i] == 0)

        # Constraint: limit number of selected to max_cover_size
        model.add_constr(xsum(x[i] for i in data.X) <= self.selection_size)

        # Objective: maximize "row" coverage (not the whole coverage of 1s)
        model.objective = maximize(xsum(is_row_covered[row] for row in data.rows))

        # Solve
        model.verbose = False
        model.optimize()

        # Solution
        selected = [i for i in data.X if float(x[i].x) >= 0.99]

        if self.verbose:
            print("=" * 40)
            print("MAX COVER OBJECTIVE:", model.objective_value)
            print("NUM ROWS COVERED:", num_row_covered.x,
                  "coverage: {:.2f}".format(num_row_covered.x / data.matrix.shape[0]))
            print("SIZE:", len(selected),
                  "reduction: {:.2f}".format((data.matrix.shape[1] - len(selected)) / data.matrix.shape[1]))
            print("SELECTED:", selected)
            print("STATUS:", model.status)
            print("=" * 40)

        assert model.status == OptimizationStatus.OPTIMAL

        # Return
        return selected

    @staticmethod
    def _validate_args(selection_size):
        if selection_size is not None:
            if selection_size <= 0:
                raise ValueError("Selection size must be greater than zero.")


# TODO not sure if this should use _BaseDispatcher or not.
class _TextBased(_BaseSupervisedSelector):
    """
    The fit() method is responsible for featurization and optimization.
    The transform() method is responsible for selecting features/content.
    """

    def __init__(self, seed: int,
                 num_features: Num,
                 featurization_method: TextWiser,
                 optimization_method: str,
                 cost_metric: str,
                 trials: int = 10):

        # Call constructor of parent class _BaseSupervisedSelector
        super().__init__(seed)

        # Set input arguments
        self.num_features = num_features
        self.featurization_method = featurization_method
        self.selection_size = num_features
        self.optimization_method = optimization_method
        self.cost_metric = cost_metric
        self.trials = trials

        # Class variables
        self.content_selector = ContentSelector(selection_size=num_features,
                                                seed=self.seed, trials=trials, verbose=False)

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> NoReturn:

        # Call fit method from parent class _BaseSupervisedSelector
        super().fit(data, labels)

        selected_indexes = self.content_selector.run(data, labels,
                                                     self.selection_size,
                                                     self.featurization_method,
                                                     self.optimization_method,
                                                     self.cost_metric)

        # Set absolute score for each feature
        abs_scores = np.zeros(data.shape[1], dtype=bool)
        abs_scores[selected_indexes] = True
        self.abs_scores = abs_scores

        # Set number of selected features
        self.num_features = len(selected_indexes)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        # Select top-k from data based on abs_scores and num_features
        return self.get_top_k(data, self.abs_scores)


# def plot_selection(name: str, embedding: Union[List[List], np.ndarray], selected: List,
#                    n_clusters: int = None, kmeans_n_init: int = 100, seed: int = 123456,
#                    selection_c: str = 'blue', centroid_marker: str = 'x', centroid_c: str = 'r',
#                    centroid_marker_s: int = 100, figsize: tuple = (10, 6), save_fig_name: str = None,
#                    **kwargs) -> NoReturn:
#     # Make scatter plot of selected content using a 2D embedding of content.
#     #
#     # Parameters
#     # ----------
#     # name: str
#     #     Name to include in plot title.
#     # embedding: Union[List[List], np.ndarray]
#     #     2-D embedding for each content item created using T-SNE, UMAP or similar.
#     # selected: List
#     #     Indices of selected content.
#     # n_clusters: int, default=None
#     #     Number of K-means clusters to fit. Cluster centroids are overlayed on scatter plot if not None.
#     # kmeans_n_init: int, default=100
#     #     Number of times the K-means algorithm will be run with different centroid seeds.
#     # seed: int, default=123456
#     #     Random seed.
#     # selection_c: str, default='blue'
#     #     Color of selected items.
#     # centroid_marker: str, default='x'
#     #     Marker of cluster centroids.
#     # centroid_c: str, default='r'
#     #     Color of cluster centroids markers.
#     # centroid_marker_s: int, default=100
#     #     Size of cluster centroid markers.
#     # figsize: tuple, default=(10, 6)
#     #     Size of figure.
#     # save_fig_name: str, default=None
#     #     Path of saved figure.
#     # **kwargs
#     #     Other parameters passed to ``matplotlib.plt.scatter``.
#     #
#     # Returns
#     # -------
#     # ax : matplotlib.axes.Axes
#     #     The scatter plot with selection.
#     #
#
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_title("PLOT: " + name + " SIZE: " + str(len(selected)) + " [blue: selected, crosses: centroids]")
#
#     # Convert embedding to numpy array
#     embedding = np.asarray(embedding)
#
#     # Plot selected points
#     mask = np.asarray([True if i in selected else False for i in range(len(embedding))])
#     ax.scatter(embedding[mask, 0], embedding[mask, 1], c=selection_c, **kwargs)
#     ax.scatter(embedding[~mask, 0], embedding[~mask, 1], c="black", **kwargs)
#
#     # Plot centroids of embeddings learned by KMeans
#     if n_clusters is not None:
#         kmeans_on_embedding = KMeans(n_clusters=n_clusters, random_state=seed, n_init=kmeans_n_init)
#         kmeans_on_embedding.fit(embedding)
#         embedding_centroids = kmeans_on_embedding.cluster_centers_
#         ax.scatter(embedding_centroids[:, 0], embedding_centroids[:, 1],
#                    s=centroid_marker_s, marker=centroid_marker, c=centroid_c)
#
#     if save_fig_name is not None:
#         plt.savefig(save_fig_name)
#
#     return ax


# def _process_data(input_df: pd.DataFrame, categories: List[str], feature_column: str, selection_size: int) \
#         -> Tuple[np.ndarray, np.ndarray]:
#     # Get label for each row based on input categories
#     labels_list = []
#     for index, row in input_df.iterrows():
#         labels = []
#         for c in categories:
#             l = c + " " + str(row[c]).replace("\n", " ")
#             labels.append(l)
#         labels_list.append(" | ".join(labels))
#     input_df["labels"] = labels_list
#
#     # Matrix
#     matrix = (input_df.labels.str.split('|', expand=True)
#                    .stack()
#                    .str.get_dummies()
#                    .sum(level=0)).T.values
#
#     num_rows, num_cols = matrix.shape
#
#     features = np.array([eval(l) if isinstance(l, str) else l for l in input_df[feature_column].tolist()])
#
#     assert (matrix.ndim == 2), "Process Data Error: matrix should 2D"
#     assert (len(features) == num_cols), \
#         f"Process Data Error: features size ({len(features)}) " \
#         f"should match the number of columns ({num_cols})"
#     assert (selection_size <= num_cols), "Process Data Error: selection_size cannot exceed num columns"
#
#     return matrix, features
