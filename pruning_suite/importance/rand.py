import random
from typing import Callable

import pruning_suite.common as c
from pruning_suite.common import PruningDataset, NAMED_MODULES, NAMED_RATIO, NAMED_IMPORTANCE


class RandomImportance(c.GenericImportance):
    def __init__(self, extract_weights_fn: dict[str, Callable]):
        self.extract_weights_fn = extract_weights_fn

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
                feature_size = len(c.select_values(w, pruned))
                importance = [random.random() for _ in range(feature_size)]
                upper_limit = max(importance) + 1
                named_features_ranking[name] = c.fill_gaps(importance, pruned, fill_value=upper_limit)

            modules_feature_importance[m] = named_features_ranking

        return modules_feature_importance
