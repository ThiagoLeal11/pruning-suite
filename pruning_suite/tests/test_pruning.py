from unittest import TestCase
import torch

from pruning_suite.pruning import binarize


class Test(TestCase):
    def test_binarize(self):
        x = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        out = binarize(0.5, x)
        values = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertListEqual(out.tolist(), values.tolist())

        out = binarize(0.6, x)
        values = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertListEqual(out.tolist(), values.tolist())

        x = torch.tensor([0, 0, 0, 1, 1, 1])
        out = binarize(0.33, x)
        self.assertEqual(out.sum(), 2)

        x = torch.tensor([0.36, 0.40, 0.51, 0.47, 0.57, 0.60])
        out = binarize(0.5, x)
        self.assertEqual(out.tolist(), [1, 1, 0, 1, 0, 0])


