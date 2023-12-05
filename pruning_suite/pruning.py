from dataclasses import dataclass
from typing import Callable, IO

import torch

import pruning_suite.common as c
from pruning_suite.common import full_class_name, get_ratio, dehydrate_named_feature


@dataclass
class FeaturesData:
    x: dict[c.MODULE, IO]
    y: c.TENSOR


# EXTRACT_FEATURES_CALLABLE = Callable[[any, torch.tensor], dict[str, torch.tensor]]
# RANKING_FUNCTION = Callable[[any, dict[str, torch.tensor], RankingData], dict[str, torch.tensor]]


class Pruning:
    def __init__(self,
                 model,
                 inputs: c.PruningDataset,
                 pruning_ratio: float | c.NAMED_RATIO,
                 extract_fn: dict[str, Callable],
                 ranking_fn: c.GenericImportance,
                 prune_fn: Callable,
                 interactive_steps: int = None,
                 global_pruning: bool = False,
                 device: str = 'cpu',
                 ):

        self.inputs = inputs
        self.model = model

        # Get a tensor of shape (batch, features, values)
        self.extract_features_map = extract_fn
        self.ranking_fn = ranking_fn
        self.prune_fn = prune_fn
        self.to_prune_modules = self.select_modules_to_prune()
        self.global_pruning = global_pruning

        self.pruning_ratio = pruning_ratio
        self.interactive_steps = interactive_steps

    def select_modules_to_prune(self):
        prune_modules = {}
        for m in self.model.modules():
            name = full_class_name(m)
            if name in self.extract_features_map:
                prune_modules[m] = name

        return prune_modules

    def wrap_forward(self, to_prune_modules: dict[any, str]):
        for m, name in to_prune_modules.items():
            extract_features_fn = self.extract_features_map[name]
            m.extract_feature_middleware = extract_features_fn.__get__(m, type(m))

            m.original_forward = m.forward.__get__(m, type(m))
            m.forward = dehydrate_named_feature.__get__(m, type(m))

    def unwrap_forward(self):
        self.clean_output()
        for m in self.model.modules():
            if hasattr(m, 'original_forward'):
                m.forward = m.original_forward
                del m.original_forward

    def clean_output(self):
        for m in self.model.modules():
            if hasattr(m, 'features_output'):
                del m.features_output

    # TODO: test
    @staticmethod
    def _get_prune_features(is_global: bool, ratio: float | c.NAMED_RATIO, named_features_importance: c.NAMED_IMPORTANCE) -> c.NAMED_IMPORTANCE:
        # print('>> Pruning features')
        prune_features = {}
        # TODO: Global pruning separated by module.
        if not is_global:
            for m, named_rank in named_features_importance.items():
                named_features = {}
                # print(f' - Pruning {full_class_name(m)} module')
                for name, rank in named_rank.items():
                    r = get_ratio(ratio, [full_class_name(m), name])
                    named_features[name] = binarize(r, rank)
                    # print(f'   - Pruning {name} features {named_features[name]}')
                prune_features[m] = named_features
            return prune_features

        # Global pruning
        all_features = []
        for named_rank in named_features_importance.values():
            for _, rank in named_rank.items():
                all_features += rank

        features = binarize(ratio, all_features)

        # Reconstruct the features for each module
        last_idx = 0
        for m, named_rank in named_features_importance.items():
            named_features = {}
            # print(f' - Pruning {full_class_name(m)} module')
            for name, rank in named_rank.items():
                named_features[name] = features[last_idx:last_idx+len(rank)]
                # print(f'   - Pruning {name} features {named_features[name]}')
                last_idx += len(rank)
            prune_features[m] = named_features

        return prune_features

    def prune_modules(self, prune_features: c.NAMED_IMPORTANCE):
        for m, to_prune in prune_features.items():
            if not to_prune:
                continue
            self.prune_fn(m, to_prune)
            c.update_pruned_features(m, to_prune)

    def prune_step(self, prune_ratio: float | c.NAMED_RATIO):
        named_features_importance = self.ranking_fn.eval_features(
            self.model, self.inputs, self.to_prune_modules, prune_ratio
        )
        prune_features = self._get_prune_features(self.global_pruning, prune_ratio, named_features_importance)
        self.prune_modules(prune_features)

    def adjust_prune_ratio(self, prune_ratio: float | c.NAMED_RATIO, steps: int):
        if isinstance(prune_ratio, (int, float)):
            return float(prune_ratio) / float(steps)

        for name, ratio in prune_ratio.items():
            prune_ratio[name] = self.adjust_prune_ratio(ratio, steps)

        return prune_ratio

    def prune(self):
        self.wrap_forward(self.to_prune_modules)

        # Prepare the model
        self.model.eval()
        with torch.no_grad():
            step = 0

            prune_ratio = self.adjust_prune_ratio(self.pruning_ratio, self.interactive_steps)
            while step < self.interactive_steps:
                print(f'>> Pruning step ({step+1}/{self.interactive_steps})')
                self.prune_step(prune_ratio)
                step += 1
                yield step, prune_ratio

        # Clean the model
        self.unwrap_forward()
        self.clean_output()

        print('>> Pruning finished')
        self.print_pruned_features()

    def print_pruned_features(self):
        for m in self.to_prune_modules.keys():
            if hasattr(m, 'features_pruned'):
                print(f'Pruned features for {full_class_name(m)}:')
                for name, features in m.features_pruned.items():
                    print(f' - {name}: {features}')


def binarize(ratio: float, rank: torch.tensor) -> list[int]:
    expected_pruned = int(round(len(rank) * ratio, 1))
    ordered = sorted([(i, x) for i, x in enumerate(rank)], key=lambda x: x[1])
    indexes = [0] * len(rank)

    while sum(indexes) < expected_pruned:
        weakest_idx = ordered.pop(0)[0]
        indexes[weakest_idx] = 1

    return indexes
