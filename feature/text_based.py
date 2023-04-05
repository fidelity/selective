# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

from typing import NoReturn, Union, List, Optional

import random
import pandas as pd
import numpy as np
from feature.base import _BaseSupervisedSelector
from feature.utils import Num
import matplotlib.pyplot as plt

from mip import Model, xsum, minimize, maximize, BINARY, INTEGER, OptimizationStatus
from sklearn.cluster import KMeans
from scipy import sparse


from textwiser import TextWiser, Embedding, Transformation




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
        assert (cost is not None), "Data Error: cost cannot be none"
        assert (matrix is not None), "Data Error: matrix cannot be none"
        assert (matrix.ndim == 2), "Data Error: matrix should 2D"
        assert (len(cost) == matrix.shape[1]), \
            "Data Error: cost vector size should match the number of columns " + \
            str(len(cost)) + " vs. " + str(matrix.shape[1])


class ContentSelector:
    """ Content Selector.

    This class contains a content selection algorithm (max_cover) that maximizes the diversity among the selected
    content, while minimizing the size of the selection that can cover all (or a maximum subset) of a set of
    predefined labels belonging to the content. The approach is closely related to set covering and is solved using a
    multi-level optimization framework.

    The class also includes baseline implementations that selects content based on a KMeans clustering of the content
    features (kmeans) or simple random selection (random).

    Attributes
    ----------
    selection_size: int
        Maximum number of content to select.
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

    """

    def __init__(self,  selection_size: int, seed: int = 123546, trials: int = 100, verbose: int = 0):
        """ContentSelector

       Creates a content selector object.

       Parameters
       ----------
        selection_size: int
            Maximum number of content to select.
        seed: int, default=123456
            Random seed.
        trials: int, default=100
            Number of trials to run for random functions to estimate average coverage.
        verbose: int, default=0
            Print intermediate results if 1.

        """

        # Validate arguments
        self._validate_args(selection_size)

        # Set input arguments
        self.selection_size = selection_size
        self.seed = seed
        self.trials = trials
        self.verbose = verbose == 1



        # Initialize class variables
        self.matrix = None
        self.features = None
        self._num_rows = None
        self._num_cols = None

    def run_content_selection(self,
                              input_df: pd.DataFrame,
                              categories: List[str], feature_column: str, featurization_method: TextWiser,
                              method: str = "max_cover",
                              cost_metric: str = "unicost") -> List:
        """Run content selection algorithm.

        Parameters
        ----------
        input_df: pd.DataFrame
            Input data frame with categories and features of content to select from.
        categories: List[str]
            List of columns in data that contains categories/labels to be covered.
        feature_column: st
            Column with numeric featurization of content text.
        method: str, default="max_cover"
            Method used to perform content selection. Supported options are:
            - "max_cover" maximizes content diversity and minimizes the number of content to cover all content labels.
            - "kmeans" clusters the content into selection_size number of clusters and then selects the items closest
              to the centroid of each of the clusters. This method does not consider the content categories/labels.
            - "greedy" performs greedy heuristic selecting items with max unit coverage until all items are covered
            - "random" performs a random selection.

        Returns
        -------
        List of indices in data that are selected.
        """

        # Process data
        self.cost_metric = cost_metric
        self.featurization_method = featurization_method
        self._process_df(input_df, categories, feature_column)

        # Run multi-level set covering optimization
        if method == "max_cover":
            selected = self._select_multi_level_optimization()
        # Run KMeans selection algorithm
        elif method == "kmeans":
            selected = self._select_kmeans()
        # Run greedy heuristic
        elif method == "greedy":
            selected = self._select_greedy()
        # Run random selection algorithm
        elif method == "random":
            selected = self._select_random()
        else:
            raise NotImplementedError(f"{method} has not been implemented.")

        # Sort selection
        selected.sort()

        return selected

    def _process_df(self, input_df: pd.DataFrame, categories: List[str], feature_column: str) -> NoReturn:

        # Get label for each row based on input categories
        # labels_list = []
        # for index, row in input_df.iterrows():
        #     labels = []
        #     for c in categories:
        #         l = c + " " + str(row[c]).replace("\n", " ")
        #         labels.append(l)
        #     labels_list.append(" | ".join(labels))
        # input_df["labels"] = labels_list

        # Matrix
        # self.matrix = (input_df.labels.str.split('|', expand=True)
        #                .stack()
        #                .str.get_dummies()
        #                .sum(level=0)).T.values
        # TODO: Xin: The process of creating matix from labels, it creates a nested list of contents.
        # TODO: Dimensions do not match with correct number of rows and columns. I used
        # categories are label
        self.matrix = categories.to_numpy()
        self._num_rows, self._num_cols = self.matrix.shape

        # Content feature (featurization with textwiser)
        feature_column = self.featurization_method.fit_transform(input_df)
        # self.features = np.array([eval(l) if isinstance(l, str) else l for l in input_df[feature_column].tolist()])
        self.features = np.array([eval(l) if isinstance(l, str) else l for l in feature_column.tolist()])

        assert (self.matrix.ndim == 2), "Process Data Error: matrix should 2D"
        assert (len(self.features) == self._num_cols), \
            f"Process Data Error: features size ({len(self.features)}) " \
            f"should match the number of columns ({self._num_cols})"
        assert (self.selection_size <= self._num_cols), "Process Data Error: selection_size cannot exceed num columns"

    def _select_multi_level_optimization(self) -> List:

        if self.verbose:
            print("\nFIRST LEVEL: Solve unicost set covering")
        num_content = self._num_cols
        unicost = np.ones(num_content)
        data = Data(cost=unicost, matrix=self.matrix)
        unicost_selected = self._solve_set_cover(data)
        if self.verbose:
            print("\nSECOND LEVEL: Maximize diversity")
        # Find clusters in the embedding space
        k = len(unicost_selected)
        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=self.trials)
        distances = kmeans.fit_transform(self.features)
        diversity_cost = [np.min(distances[i]) for i in range(num_content)]

        # Scale contexts so that sum of costs remain constant
        diversity_cost = [c * num_content / sum(diversity_cost) for c in diversity_cost]

        # Update the costs
        data.cost = diversity_cost

        # Solve set covering with diversity guidance
        diversity_selected = self._solve_set_cover(data)

        if self.verbose:
            print("\nTHIRD LEVEL: Maximum Coverage of", self._num_rows, "rows by selecting",
                  self.selection_size, "columns out of", len(diversity_selected))

        # Among the most diverse columns with full coverage
        # Select a subset of given max_cover_size
        # While maximizing the coverage
        zeros = np.zeros(self._num_cols)  # max coverage has a different objective function
        data = Data(cost=zeros, matrix=self.matrix)
        selected = self._solve_max_cover(data, diversity_selected)

        if self.verbose:
            print("\nOptimized Selection:", selected)
            print("Selection size:", len(selected))

        # Return solution
        return selected

    def _select_random(self) -> List[int]:
        # it returns selected column indices

        # Set the seed
        random.seed(self.seed)

        best_selected = []
        best_covered = 0
        for t in range(self.trials):
            # Select a sample of selection_size without repetition
            selected = []
            while len(selected) < self.selection_size and self._num_cols is not None:
                i = random.randint(0, self._num_cols - 1)
                while i in selected:
                    i = random.randint(0, self._num_cols -1)
                selected.append(i)

            # selected = random.sample([i for i in range(self._num_cols)], self.selection_size)

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
            print("=" * 40)

        return best_selected

    def _select_greedy(self) -> List:

        if self.cost_metric == "diverse":
            optcost = np.ones(self._num_cols)
            k = len(optcost)
            num_content = self._num_cols
            kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=self.trials)
            distances = kmeans.fit_transform(self.features)
            # distances = kmeans.fit_transform(self.matrix)
            diversity_cost = [np.sum(distances[:,i]) for i in range(num_content)]
            # diversity_cost = [np.min(distances[:,i]) for i in range(num_content)]

            # Scale contexts so that sum of costs remain constant
            diversity_cost = [c * num_content / sum(diversity_cost) for c in diversity_cost]
            optcost = diversity_cost
        else:
            optcost = np.ones(self._num_cols)


        # Compressed sparse column (transposed for convenience)
        sparse_col = sparse.csr_matrix(self.matrix.T, copy=True)

        # Initial guess of the Lagrangian multiplier with greedy algorithm
        adjusted_cost = optcost / sparse_col.dot(np.ones(self._num_rows))
        cost_matrix = adjusted_cost * self.matrix + np.amax(adjusted_cost) * (~self.matrix)
        u = adjusted_cost[np.argmin(cost_matrix, axis=1)]

        # Nothing is selected, everything is uncovered
        selected = np.zeros(self._num_cols, dtype=bool)
        iuncovered = np.ones(self._num_rows, dtype=bool)
        score = np.zeros(self._num_cols)

        epsilon = 1E-5
        size = sum(selected)
        # While there are uncovered rows and below selection size
        while np.count_nonzero(iuncovered) > 0 and size < self.selection_size:

            # Faster than indexing, made possible by sparse_col.dot
            mu = sparse_col.dot(iuncovered.astype(int)).astype(float)
            mu[mu <= epsilon] = epsilon
            # Set Lagrange multiplier zero for covered rows
            u[~iuncovered] = 0
            gamma = (optcost - sparse_col.dot(u))
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

        if size == self.selection_size:
            print("Warning: max greedy reached selection size", self.selection_size)

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
            print("=" * 40)

        return selected

    def _select_kmeans(self) -> List:

        kmeans = KMeans(n_clusters=self.selection_size, random_state=self.seed, n_init=self.trials)
        kmeans.fit(self.features)
        selected = self._get_closest_to_centroids(kmeans)
        num_row_covered = self._get_num_row_covered(selected)

        if self.verbose:
            print("\nKMEANS SELECTION:", self.selection_size, " columns to cover rows", self._num_rows)
            print("=" * 40)
            print("SIZE:", len(selected), "reduction: {:.2f}".format((self._num_cols - len(selected)) / self._num_cols))
            print("SELECTED:", selected)
            print("NUM ROWS COVERED:", num_row_covered, "coverage: {:.2f}".format(num_row_covered / self._num_rows))
            print("STATUS: KMEANS")
            print("=" * 40)

        return selected

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

        # Solve (optimize using Gurobi)
        model.verbose = 0
        model.optimize()
        assert model.status == OptimizationStatus.OPTIMAL, "Max Cover Error: optimal solution not found."

        # Solution

        selected = [i for i in data.X if float(x[i].x) >= 0.99]
        num_row_covered = self._get_num_row_covered(selected)

        if self.verbose:
            print("=" * 40)
            print("OBJECTIVE:", model.objective_value)
            print("SIZE:", len(selected), "reduction: {:.2f}".format((self._num_cols - len(selected)) / self._num_cols))
            print("SELECTED:", selected)
            print("NUM ROWS COVERED:", num_row_covered, "coverage: {:.2f}".format(num_row_covered / self._num_rows))
            print("STATUS:", model.status)
            print("=" * 40)

        # Return
        return selected

    def _solve_max_cover(self, data: Data, selected: List) -> List:

        # If selected is given, limit the max_cover_size
        if selected is not None and len(selected) > 0:
            assert (self.selection_size <= len(selected)), "Max Cover Error: max_cover_size cannot exceed num selected"

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
        model.verbose = 0
        model.optimize()

        # Solution
        selected = [i for i in data.X if float(x[i].x) >= 0.99]

        if self.verbose:
            print("=" * 40)
            print("OBJECTIVE:", model.objective_value)
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

    def _get_num_row_covered(self, selected) -> int:
        assert self.matrix is not None, "matrix is not initialized"

        # Create indicator to count number of covered rows
        row_covered = np.sum(self.matrix[:, selected], axis=1)

        row_covered[row_covered > 0] = 1

        num_row_covered = sum(row_covered)

        return num_row_covered

    @staticmethod
    def _validate_args(selection_size):
        if selection_size <= 0:
            raise ValueError("Selection size must be greater than zero.")


def plot_selection(name: str, embedding: Union[List[List], np.ndarray], selected: List,
                   n_clusters: int = None, kmeans_n_init: int = 100, seed: int = 123456,
                   selection_c: str = 'blue', centroid_marker: str = 'x', centroid_c: str = 'r',
                   centroid_marker_s: int = 100, figsize: tuple = (10, 6), save_fig_name: str = None,
                   **kwargs) -> NoReturn:
    """Make scatter plot of selected content using a 2D embedding of content.

    Parameters
    ----------
    name: str
        Name to include in plot title.
    embedding: Union[List[List], np.ndarray]
        2-D embedding for each content item created using T-SNE, UMAP or similar.
    selected: List
        Indices of selected content.
    n_clusters: int, default=None
        Number of K-means clusters to fit. Cluster centroids are overlayed on scatter plot if not None.
    kmeans_n_init: int, default=100
        Number of times the K-means algorithm will be run with different centroid seeds.
    seed: int, default=123456
        Random seed.
    selection_c: str, default='blue'
        Color of selected items.
    centroid_marker: str, default='x'
        Marker of cluster centroids.
    centroid_c: str, default='r'
        Color of cluster centroids markers.
    centroid_marker_s: int, default=100
        Size of cluster centroid markers.
    figsize: tuple, default=(10, 6)
        Size of figure.
    save_fig_name: str, default=None
        Path of saved figure.
    **kwargs
        Other parameters passed to ``matplotlib.plt.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The scatter plot with selection.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("PLOT: " + name + " SIZE: " + str(len(selected)) + " [blue: selected, crosses: centroids]")

    # Convert embedding to numpy array
    embedding = np.asarray(embedding)

    # Plot selected points
    mask = np.asarray([True if i in selected else False for i in range(len(embedding))])
    ax.scatter(embedding[mask, 0], embedding[mask, 1], c=selection_c, **kwargs)
    ax.scatter(embedding[~mask, 0], embedding[~mask, 1], c="black", **kwargs)

    # Plot centroids of embeddings learned by KMeans
    if n_clusters is not None:
        kmeans_on_embedding = KMeans(n_clusters=n_clusters, random_state=seed, n_init=kmeans_n_init)
        kmeans_on_embedding.fit(embedding)
        embedding_centroids = kmeans_on_embedding.cluster_centers_
        ax.scatter(embedding_centroids[:, 0], embedding_centroids[:, 1],
                   s=centroid_marker_s, marker=centroid_marker, c=centroid_c)

    if save_fig_name is not None:
        plt.savefig(save_fig_name)

    return ax


# TODO not sure if this should use _BaseDispatcher or not. We can decide later
class _TextBased(_BaseSupervisedSelector):
    """
    featurization is done inb 'fit' function and optimization is done in 'transform' function
    """

    def __init__(self, seed: int, num_features: Num,
                 featurization_method: TextWiser, optimization_method: str, cost_metric: Optional[str] = None):
        # Call constructor of parent class _BaseSupervisedSelector
        super().__init__(seed)

        self.num_features = num_features    # this could be int or float
        self.featurization_method = featurization_method
        self.optimization_method = optimization_method
        self.cost_metric = cost_metric
        self.content_selector = ContentSelector(selection_size=num_features, seed=42, trials=100, verbose=True)

    def fit(self, data: pd.DataFrame, labels: Union[pd.Series, pd.DataFrame]) -> NoReturn:
        # print("FIT: ", self.num_features, self.featurization_method, self.optimization_method, self.cost_metric)

        # Call fit method from parent class _BaseSupervisedSelector
        super().fit(data, labels)

        # Get the text columns dynamically
        text_columns = [col for col in data.columns if col.startswith("item")]

        # Perform content selection using the specified method
        selected_indicies = self.content_selector.run_content_selection(data, labels, text_columns,
                                                                        self.featurization_method,
                                                                        self.optimization_method, self.cost_metric)

        # Only select the features that were selected during fit
        selected_features = [col for i, col in enumerate(text_columns) if i in selected_indicies]
        self.selected_features_ = selected_features



    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        print("TRANSFORM: ", self.num_features, self.featurization_method, self.optimization_method, self.cost_metric)
        # TODO Solve? or setup? appropriate set cover problem with given parameters

        # TODO return the features accordingly
        return -1
