# Optimized Item Selection Datasets

We provide the datasets that are used to test the multi-level optimization framework ([CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'22](https://arxiv.org/abs/2112.03105)), for solving Item Selection Problem (ISP) to boost exploration in Recommender Systems.

## Overview of Datasets
The datasets include:

* [**GoodReads datasets**](book_recommenders_data/) for book recommenders. Two datasets are randomly selected from the source data [GoodReads Book Reviews](https://dl.acm.org/doi/10.1145/3240323.3240369), a small version with 1000 items and a large version with 10,000 items. For book recommendations, there are 11 different genres (e.g., fiction, non-fiction, children), 231 different publishers (e.g. Vintage, Penguin Books, Mariner Books), and genre-publisher pairs. This leads to 574 and 1,322 unique book labels for the small and large datasets, respectively.

* [**MovieLens datasets**](movie_recommenders_data/) for movie recommenders. Two datasets are randomly selected from the source data [MovieLens Movie Ratings](https://dl.acm.org/doi/10.1145/2827872), a small version with 1000 items and a large version with 10,000 items. For movie recommendations, there are 19 different genres (e.g. action, comedy, drama, romance), 587 different producers, 34 different languages (e.g. English, French, Mandarin), and genre-language pairs. This leads to 473 and 1,011 unique movie labels for the small and large datasets, respectively.

Each dataset in GoodReads and MovieLens contains a `*_data.csv` file, which contain the text content (i.e., title + description) of the items, and a `*_label.csv`, which contains the labels (e.g., genre or language) and a binary 0/1 denoting whether an item exbihits a label. 

Each column in the csv file is for an item, indexed by book/movie ID. The order of columns in data and label files are the same.

[Selective](https://github.com/fidelity/selective) implements the multi-objective optimization approach from ([CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'22](https://arxiv.org/abs/2112.03105)) as part of `TextBased Selection`. 

By solving the ISP with Text-based Selection in Selective, we select a smaller subset of items with maximum diversity in the latent embedding space of items and maximum coverage of labels.

## Usage Example
```python
# Import Selective (for text-based selection) and TextWiser (for embedding space)
import pandas as pd
from feature.selector import Selective, SelectionMethod
from textwiser import TextWiser, Embedding, Transformation

# Load Text Contents
data = pd.read_csv("goodreads_1k_data.csv").astype(str)

# Load Labels 
labels = pd.read_csv("goodreads_1k_label.csv")
labels.set_index('label', inplace=True)

# TextWiser featurization method to create text embeddings
textwiser = TextWiser(Embedding.TfIdf(), Transformation.NMF(n_components=20, random_state=1234))

# Text-based selection
selector = Selective(SelectionMethod.TextBased(num_features=30, featurization_method=textwiser))

# Result
subset = selector.fit_transform(data, labels)
print("Reduction:", list(subset.columns))
```

## Citation
If you use ISP in our research/applications, please cite as follows: