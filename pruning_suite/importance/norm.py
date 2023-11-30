from functools import reduce
from typing import Callable

import pruning_suite.common as c
from pruning_suite.common import PruningDataset, NAMED_MODULES, NAMED_RATIO, NAMED_IMPORTANCE


class NormBasedImportance(c.GenericImportance):
    def __init__(self, extract_weights_fn: dict[str, Callable], norm: str = 'l1', scale: bool = False):
        self.norm = norm
        self.extract_weights_fn = extract_weights_fn
        self.scale = scale

    @staticmethod
    def _get_norm(weights: c.TENSOR, norm: str, scale=False) -> c.TENSOR:
        dims = list(range(1, weights.dim()))
        n_values = reduce(lambda x, y: x * y, weights.shape[1:])

        if scale:
            weights = weights / (float(n_values) / 10.0)

        if norm == 'l1':
            return weights.abs().sum(dim=dims)
        elif norm == 'l2':
            return weights.pow(2).sum(dim=dims).sqrt()
        elif norm == 'linf':
            return weights.abs().max(dim=dims)
        else:
            raise Exception(f'Norm {norm} not found')

    def eval_features(self, model, data: PruningDataset, to_prune_modules: NAMED_MODULES,
                      prune_ratio: NAMED_RATIO) -> NAMED_IMPORTANCE:
        modules_feature_importance = {}
        for m in to_prune_modules.keys():
            extract_fn = self.extract_weights_fn[c.full_class_name(m)]
            weights = extract_fn(m)
            modules_feature_importance[m] = {
                n: self._get_norm(w, self.norm, self.scale)
                for n, w in weights.items()
            }

        return modules_feature_importance
