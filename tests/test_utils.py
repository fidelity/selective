# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: GNU GPLv3

import numpy as np
from sklearn.datasets import load_iris
from feature.utils import get_data_label, reduce_memory, DataTransformer

from tests.test_base import BaseTest


class TestUtils(BaseTest):

    def test_mem_usage(self):
        data, label = get_data_label(load_iris())
        data_reduced = reduce_memory(data, verbose=False)
        self.assertEqual(data_reduced.shape, (150, 4))
        self.assertFalse(data_reduced.isna().any().any())

    def test_cap_floor(self):
        data, label = get_data_label(load_iris())

        # Fit transformer and transform to numeric contexts
        data_transformer = DataTransformer()
        contexts = data_transformer.fit(data)
        contexts = data_transformer.transform(data)
        contexts = data_transformer.fit_transform(data)
        self.assertFalse(np.isnan(contexts).any())
        self.assertEqual(contexts.shape, (150, 4))
