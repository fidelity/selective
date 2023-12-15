# Optimized Item Selection Datasets

We provide the datasets that are used to test the multi-level optimization framework ([CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'22](https://arxiv.org/abs/2112.03105)), for solving Item Selection Problem (ISP) to boost exploration in Recommender Systems. The datasets are extracted and processed from original public sources that will be introduced in the following, and are only used in our research study in ISP. Please kindly refrain from distributing the datasets.

## Overview of Datasets
The datasets include:

* [**GoodReads datasets**](book_recommenders_data/) for book recommenders. Two datasets are randomly selected from the source data [GoodReads Book Reviews](https://dl.acm.org/doi/10.1145/3240323.3240369), a small version with 1000 items and a large version with 10,000 items. For book recommendations, there are 11 different genres (e.g., fiction, non-fiction, children), 231 different publishers (e.g. Vintage, Penguin Books, Mariner Books), and genre-publisher pairs. This leads to 574 and 1,322 unique book labels for the small and large datasets, respectively.

* [**MovieLens datasets**](movie_recommenders_data/) for movie recommenders. Two datasets are randomly selected from the source data [MovieLens Movie Ratings](https://dl.acm.org/doi/10.1145/2827872), a small version with 1000 items and a large version with 10,000 items. For movie recommendations, there are 19 different genres (e.g. action, comedy, drama, romance), 587 different producers, 34 different languages (e.g. English, French, Mandarin), and genre-language pairs. This leads to 473 and 1,011 unique movie labels for the small and large datasets, respectively.

Each dataset in GoodReads and MovieLens contains a `*_data.csv` file, which contain the text content (i.e., title + description) of the items, and a `*_label.csv`, which contains the labels (e.g., genre or language) and a binary 0/1 denoting whether an item exbihits a label. 

Each column in the csv file is for an item, indexed by book/movie ID. The order of columns in data and label files are the same.

[Selective](https://github.com/fidelity/selective) implements the multi-objective optimization approach from ([CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'22](https://arxiv.org/abs/2112.03105)) as part of `TextBased Selection`. 

By solving the ISP with Text-based Selection in Selective, we select a smaller subset of items with maximum diversity in the latent embedding space of items and maximum coverage of labels.

## Quick Start
To run the example, install required packages by `pip install selective datasets`
```python
# Import Selective (for text-based selection) and TextWiser (for embedding space)
import pandas as pd
from datasets import load_dataset
from textwiser import TextWiser, Embedding, Transformation
from feature.selector import Selective, SelectionMethod


# Load Text Contents
data = load_dataset('skadio/optimized_item_selection', data_files='book_recommenders_data/goodreads_1k_data.csv', split='train')
data = data.to_pandas()

# Load Labels 
labels = load_dataset('skadio/optimized_item_selection', data_files='book_recommenders_data/goodreads_1k_label.csv', split='train')
labels = labels.to_pandas()
labels.set_index('label', inplace=True)

# TextWiser featurization method to create text embeddings
textwiser = TextWiser(Embedding.TfIdf(), Transformation.NMF(n_components=20, random_state=1234))

# Text-based selection with the default selection method - Multi-Level Optimization that maximizes 
# coverage and diversity within an upper bound on subset size [CPAIOR'21, DSO@IJCAI'22], by choosing 
# the default configuration, i.e. optimization_method="exact" and cost_metric ="diverse".
selector = Selective(SelectionMethod.TextBased(num_features=30, featurization_method=textwiser))

# Result
subset = selector.fit_transform(data, labels)
print("Reduction:", list(subset.columns))
```
## Advanced Usages
Text-based Selection provides access to multiple selection methods. The following configurations are 
available to apply these methods:

- (Default) Solve for Problem *P_max_cover@t* in **CPAIOR'21** - Selecting a subset of items that 
maximizes coverage of labels and maximizes the diversity in latent embedding space within an upper 
bound on subset size.
```python
selector = Selective(SelectionMethod.TextBased(num_features=30, 
                                               featurization_method=textwiser,
                                               optimization_method='exact', 
                                               cost_metric='diverse'))
```
- Solve for Problem *P_unicost* in **CPAIOR'21** - Selecting a subset of items that covers all labels.
```python
selector = Selective(SelectionMethod.TextBased(num_features=None, 
                                               optimization_method='exact', 
                                               cost_metric='unicost'))
```
- Solve for Problem *P_diverse* in **CPAIOR'21** - Selecting a subset of items with maximized diversity 
in the latent embedding space while still maintaining the coverage over all labels.
```python
selector = Selective(SelectionMethod.TextBased(num_features=None,
                                               featurization_method=textwiser, 
                                               optimization_method='exact', 
                                               cost_metric='diverse'))
```
- Selecting a subset of items that only maximizes coverage within an upper bound on subset size.
```python
selector = Selective(SelectionMethod.TextBased(num_features=30, 
                                               optimization_method='exact', 
                                               cost_metric='unicost'))
```
- Selecting a subset by performing random selection. If num_features is not set, subset size is defined 
by solving #2.
```python
selector = Selective(SelectionMethod.TextBased(num_features=None, optimization_method='random'))
```
- Selecting a subset by performing random selection. Subset size is defined by num_features.
```python
selector = Selective(SelectionMethod.TextBased(num_features=30, 
                                               optimization_method='random'))
```
- Selecting a subset by adding an item each time using a greedy heuristic in selection with a given
cost_metric, i.e. `diverse` by default or `unicost`. If num_features is not set, subset size is defined 
by solving #2.
```python
selector = Selective(SelectionMethod.TextBased(num_features=None, 
                                               optimization_method='greedy',
                                               cost_metric='unicost'))
```
- Selecting a subset by adding an item each time using a greedy heuristic in selection with a given
cost_metric, i.e. `diverse` by default or `unicost`.
```python
selector = Selective(SelectionMethod.TextBased(num_features=30,
                                               optimization_method='greedy',
                                               cost_metric='unicost'))
```
- Selecting a subset by clustering items into a number of clusters and the items close to the centroids 
are selected. If num_features is not set, subset size is defined by solving #2. `cost_metric` argument 
is not used in this method.
```python
selector = Selective(SelectionMethod.TextBased(num_features=None, optimization_method='kmeans'))
```
- Selecting a subset by clustering items into a number of clusters and the items close to the centroids 
are selected. `cost_metric` argument is not used in this method.
```python
selector = Selective(SelectionMethod.TextBased(num_features=30,
                                               optimization_method='kmeans'))
```

## Citation
If you use ISP in our research/applications, please cite as follows:
```bibtex
  @inproceedings{cpaior2021,
    title={Optimized Item Selection to Boost Exploration for Recommender Systems},
    author={Serdar Kadıoğlu and Bernard Kleynhans and Xin Wang},
    booktitle={Proceedings of Integration of Constraint Programming, Artificial Intelligence, and Operations Research: 18th International Conference, CPAIOR 2021, Vienna, Austria, July 5–8, 2021},
    url={https://doi.org/10.1007/978-3-030-78230-6_27},
    pages = {427–445},
    year={2021}
  }
```

```bibtex
@inproceedings{ijcai2022,
      title={Active Learning Meets Optimized Item Selection}, 
      author={Bernard Kleynhans and Xin Wang and Serdar Kadıoğlu},
      booktitle={The IJCAI-22 Workshop: Data Science meets Optimisation}
      publisher={arXiv},
      url={https://arxiv.org/abs/2112.03105},
      year={2022}
}
```