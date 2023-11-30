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
        elif norm == 'max':
            w = weights.abs()
            for _ in dims:
                w, _ = w.max(dim=-1)
            return w
        else:
            raise Exception(f'Norm {norm} not found')

    def eval_features(self, model, data: PruningDataset, to_prune_modules: NAMED_MODULES,
                      prune_ratio: NAMED_RATIO) -> NAMED_IMPORTANCE:
        modules_feature_importance = {}
        for m in to_prune_modules.keys():
            already_pruned = c.get_pruned_features(m)

            extract_fn = self.extract_weights_fn[c.full_class_name(m)]
            weights = extract_fn(m)

            named_features_ranking = {}
            for name, w in weights.items():
                pruned = already_pruned.get(name, None)
                importance = self._get_norm(
                    weights=c.select_values(w, pruned),
                    norm=self.norm,
                    scale=self.scale
                )
                upper_limit = w.max() + 1
                named_features_ranking[name] = c.fill_gaps(importance, pruned, fill_value=upper_limit)

            modules_feature_importance[m] = named_features_ranking

        return modules_feature_importance
