[![ci](https://github.com/fidelity/selective/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/fidelity/selective/actions/workflows/ci.yml) [![PyPI version fury.io](https://badge.fury.io/py/selective.svg)](https://pypi.python.org/pypi/selective/) [![PyPI license](https://img.shields.io/pypi/l/selective.svg)](https://pypi.python.org/pypi/selective/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Downloads](https://static.pepy.tech/personalized-badge/selective?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/selective)


# Selective: Feature Selection Library
**Selective** is a white-box feature selection library that supports supervised and unsupervised selection methods for classification and regression tasks. 

Selective also provides optimized item selection based on diversity of text embeddings (via [TextWiser](https://github.com/fidelity/textwiser)) and 
the coverage of binary labels by solving a multi-objective optimization problem ([CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'22](https://arxiv.org/abs/2112.03105)). The approach showed to speed-up online experimentation significantly and boost recommender systems [NVIDIA GTC'22](https://www.youtube.com/watch?v=_v-B2nRy79w).  

The library provides:

* Simple to complex selection methods: Variance, Correlation, Statistical, Linear, Tree-based, or Customized.
* [Text-based selection](#text-based-selection) to maximize diversity in text embeddings and metadata coverage.
* Interoperable with data frames as the input.
* Automated task detection. No need to know what feature selection method works with what machine learning task.
* Benchmarking multiple selectors using cross-validation with built-in parallelization.
* Inspection of the results and feature importance. 

Selective is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

## Quick Start
```python
# Import Selective and SelectionMethod
from sklearn.datasets import load_boston
from feature.utils import get_data_label
from feature.selector import Selective, SelectionMethod

# Data
data, label = get_data_label(load_boston())

# Feature selectors from simple to more complex
selector = Selective(SelectionMethod.Variance(threshold=0.0))
selector = Selective(SelectionMethod.Correlation(threshold=0.5, method="pearson"))
selector = Selective(SelectionMethod.Statistical(num_features=3, method="anova"))
selector = Selective(SelectionMethod.Linear(num_features=3, regularization="none"))
selector = Selective(SelectionMethod.TreeBased(num_features=3))

# Feature reduction
subset = selector.fit_transform(data, label)
print("Reduction:", list(subset.columns))
print("Scores:", list(selector.get_absolute_scores()))
```


## Available Methods

|                                                           Method                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                        Options                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:--------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [Variance per Feature](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html) |                                                                                                                                                                                                                                                                                                                                                                                                                                      `threshold`                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|   [Correlation pairwise Features](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)   |                                                                                                                                                                                                                                                                     [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) <br> [Kendall Rank Correlation Coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) <br> [Spearman's Rank Correlation Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) <br>                                                                                                                                                                                                                                                                      |
|    [Statistical Analysis](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)     |                                                                                                             [ANOVA F-test Classification](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html) <br> [F-value Regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) <br> [Chi-Square](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) <br> [Mutual Information Classification](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) <br> [Variance Inflation Factor](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)                                                                                                              |
|                             [Linear Methods](https://en.wikipedia.org/wiki/Linear_regression)                              |                                                                                                   [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression) <br> [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression) <br> [Lasso Regularization](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso) <br> [Ridge Regularization](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge) <br>                                                                                                    |
|                          [Tree-based Methods](https://scikit-learn.org/stable/modules/tree.html)                           | [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) <br> [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier) <br> [Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) <br> [XGBoost](https://xgboost.readthedocs.io/en/latest/) <br> [LightGBM](https://lightgbm.readthedocs.io/en/latest/) <br> [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) <br> [CatBoost](https://github.com/catboost)<br> [Gradient Boosting Tree](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) <br> |
|  [Text-based Methods](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27)  |                                                                                                                                                                                                                                                                                                                                              `featurization_method` = [TextWiser](https://github.com/fidelity/textwiser) <br> `optimization_method = ["exact", "greedy", "kmeans", "random"]` <br> `cost_metric = ["unicost", "diverse"]`                                                                                                                                                                                                                                                                                                                                              |



## Benchmarking

```python
# Imports
from sklearn.datasets import load_boston
from feature.utils import get_data_label
from xgboost import XGBClassifier, XGBRegressor
from feature.selector import SelectionMethod, benchmark, calculate_statistics

# Data
data, label = get_data_label(load_boston())

# Selectors
corr_threshold = 0.5
num_features = 3
tree_params = {"n_estimators": 50, "max_depth": 5, "random_state": 111, "n_jobs": 4}
selectors = {

  # Correlation methods
  "corr_pearson": SelectionMethod.Correlation(corr_threshold, method="pearson"),
  "corr_kendall": SelectionMethod.Correlation(corr_threshold, method="kendall"),
  "corr_spearman": SelectionMethod.Correlation(corr_threshold, method="spearman"),
  
  # Statistical methods
  "stat_anova": SelectionMethod.Statistical(num_features, method="anova"),
  "stat_chi_square": SelectionMethod.Statistical(num_features, method="chi_square"),
  "stat_mutual_info": SelectionMethod.Statistical(num_features, method="mutual_info"),
  
  # Linear methods
  "linear": SelectionMethod.Linear(num_features, regularization="none"),
  "lasso": SelectionMethod.Linear(num_features, regularization="lasso", alpha=1000),
  "ridge": SelectionMethod.Linear(num_features, regularization="ridge", alpha=1000),
  
  # Non-linear tree-based methods
  "random_forest": SelectionMethod.TreeBased(num_features),
  "xgboost_classif": SelectionMethod.TreeBased(num_features, estimator=XGBClassifier(**tree_params)),
  "xgboost_regress": SelectionMethod.TreeBased(num_features, estimator=XGBRegressor(**tree_params))
}

# Benchmark (sequential)
score_df, selected_df, runtime_df = benchmark(selectors, data, label, cv=5)
print(score_df, "\n\n", selected_df, "\n\n", runtime_df)

# Benchmark (in parallel)
score_df, selected_df, runtime_df = benchmark(selectors, data, label, cv=5, n_jobs=4)
print(score_df, "\n\n", selected_df, "\n\n", runtime_df)

# Get benchmark statistics by feature
stats_df = calculate_statistics(score_df, selected_df)
print(stats_df)
```

## Text-based Selection
This example shows how to use text-based selection. In this scenario, we would like to select a subset of articles that is most diverse in the text embedding space and covers a range of topics. 

```python
# Import Selective and TextWiser
import pandas as pd
from feature.selector import Selective, SelectionMethod
from textwiser import TextWiser, Embedding, Transformation

# Data with the text content of each article
data = pd.DataFrame({"article_1": ["article text here"],
                     "article_2": ["article text here"],
                     "article_3": ["article text here"],
                     "article_4": ["article text here"],
                     "article_5": ["article text here"]})

# Labels to denote 0/1 coverage metadata for each article 
# across four labels, e.g., sports, international, entertainment, science    
labels = pd.DataFrame({"article_1": [1, 1, 0, 1],
                       "article_2": [0, 1, 0, 0],
                       "article_3": [0, 0, 1, 0],
                       "article_4": [0, 0, 1, 1],
                       "article_5": [1, 1, 1, 0]},
                      index=["label_1", "label_2", "label_3", "label_4"])

# TextWiser featurization method to create text embeddings
textwiser = TextWiser(Embedding.TfIdf(), Transformation.NMF(n_components=20))

# Text-based selection
# The goal is to select a subset of articles 
# that is most diverse in the text embedding space of articles
# and covers the most labels in each topic
selector = Selective(SelectionMethod.TextBased(num_features=2, featurization_method=textwiser))

# Feature reduction
subset = selector.fit_transform(data, labels)
print("Reduction:", list(subset.columns))
```

## Visualization

```python
import pandas as pd
from sklearn.datasets import load_boston
from feature.utils import get_data_label
from feature.selector import SelectionMethod, Selective, plot_importance

# Data
data, label = get_data_label(load_boston())

# Feature Selector
selector = Selective(SelectionMethod.Linear(num_features=10, regularization="none"))
subset = selector.fit_transform(data, label)

# Plot Feature Importance
df = pd.DataFrame(selector.get_absolute_scores(), index=data.columns)
plot_importance(df)
```

## Installation

Selective requires **Python 3.6+** and can be installed from PyPI using ``pip install selective``.

## Source 

Alternatively, you can build a wheel package on your platform from scratch using the source code:

```bash
git clone https://github.com/fidelity/selective.git
cd selective
pip install setuptools wheel # if wheel is not installed
python setup.py sdist bdist_wheel
pip install dist/selective-X.X.X-py3-none-any.whl
```

## Test your setup

```
cd selective
python -m unittest discover tests
```

## Support

Please submit bug reports and feature requests as [Issues](https://github.com/fidelity/selective/issues).

## License
Selective is licensed under the [GNU GPL 3.0.](https://github.com/fidelity/selective/blob/master/LICENSE)
