import tempfile
from dataclasses import dataclass
from typing import Callable, IO

import torch
from torch.utils.data import DataLoader


MODULE = any
TENSOR = torch.tensor
NAMED_FEATURES = dict[str, TENSOR]
NAMED_FEATURE_IMPORTANCE = dict[MODULE, NAMED_FEATURES]
NAMED_RATIO = dict[str, float | dict[str, float]]


@dataclass
class PruningDataset:
    train: DataLoader
    test: DataLoader


@dataclass
class RankingData:
    train_x: TENSOR
    train_y: TENSOR
    test_x: TENSOR
    test_y: TENSOR


@dataclass
class FeaturesData:
    x: dict[MODULE, IO]
    y: TENSOR


# EXTRACT_FEATURES_CALLABLE = Callable[[any, torch.tensor], dict[str, torch.tensor]]
# RANKING_FUNCTION = Callable[[any, dict[str, torch.tensor], RankingData], dict[str, torch.tensor]]


class Pruning:
    def __init__(self,
                 model,
                 inputs: PruningDataset,
                 pruning_ratio: float | NAMED_RATIO,
                 extract_fn: dict[str, Callable],
                 ranking_fn: Callable,
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
            m.forward = store_feature_output.__get__(m, type(m))

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

    @staticmethod
    # TODO: test
    def _reshape_data(x: NAMED_FEATURES) -> NAMED_FEATURES:
        data = {}
        for name, features in x.items():
            batch, num_features = features.shape[0], features.shape[1]
            d = features.reshape(batch, num_features, -1).detach()
            data[name] = d
        return data

    @staticmethod
    def hydrate_named_features(x: IO) -> NAMED_FEATURES:
        return torch.load(x.name)

    def eval_features(self, train: FeaturesData, test: FeaturesData, to_prune_modules, prune_ratio) -> NAMED_FEATURE_IMPORTANCE:
        modules_feature_importance = {}

        # Make value importance ranking
        for idx, m in enumerate(to_prune_modules.keys()):
            print(f' - {full_class_name(m)} module ({idx+1}/{len(to_prune_modules)})')
            x_train = self._reshape_data(self.hydrate_named_features(train.x[m]))
            x_test = self._reshape_data(self.hydrate_named_features(test.x[m]))

            named_features_ranking = {}
            for name in x_train.keys():
                r = self.get_ratio(prune_ratio, [full_class_name(m), name])
                if round(r, 4) <= 0.0001:
                    continue  # Skip importance for layer without pruning

                print(f'   - {name} ({x_train[name].shape[1]} features)')
                rank = self.ranking_fn(x_train[name], train.y, x_test[name], test.y)

                # Update ranking to consider already pruned features
                if hasattr(m, 'features_pruned'):
                    features_pruned = m.features_pruned[name]
                    rank = [x if not is_pruned else 1 for x, is_pruned in zip(rank, features_pruned)]

                named_features_ranking[name] = rank
            modules_feature_importance[m] = named_features_ranking

        return modules_feature_importance

    def get_ratio(self, ratio_value: float | NAMED_RATIO, names: list[str]) -> float:
        if isinstance(ratio_value, float):
            return ratio_value

        if len(names) == 0:
            raise Exception('Missing name for ratio')

        name = names.pop(0)
        return self.get_ratio(ratio_value.get(name, 0), names)

    # TODO: test
    def _get_prune_features(self, is_global: bool, ratio: float | NAMED_RATIO, named_features_importance: NAMED_FEATURE_IMPORTANCE) -> NAMED_FEATURE_IMPORTANCE:
        print('>> Pruning features')
        prune_features = {}
        if not is_global:
            for m, named_rank in named_features_importance.items():
                named_features = {}
                print(f' - Pruning {full_class_name(m)} module')
                for name, rank in named_rank.items():
                    r = self.get_ratio(ratio, [full_class_name(m), name])
                    named_features[name] = binarize(r, rank)
                    print(f'   - Pruning {name} features {named_features[name]}')
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
            print(f' - Pruning {full_class_name(m)} module')
            for name, rank in named_rank.items():
                named_features[name] = features[last_idx:last_idx+len(rank)]
                print(f'   - Pruning {name} features {named_features[name]}')
                last_idx += len(rank)
            prune_features[m] = named_features

        return prune_features

    def extract_features(self, x, y):
        self.model(x)
        data = FeaturesData(
            x={
                m: m.features_output
                for m in self.to_prune_modules.keys()
            },
            y=y
        )
        self.clean_output()
        return data

    def prune_modules(self, prune_features: NAMED_FEATURE_IMPORTANCE):
        for m, features in prune_features.items():
            if not features:
                continue
            self.prune_fn(m, features)

        # Persist features pruned.
        self.sum_pruned_features(prune_features)

    def sum_pruned_features(self, prune_features: NAMED_FEATURE_IMPORTANCE):
        for m, features in prune_features.items():
            if not hasattr(m, 'features_pruned'):
                m.features_pruned = features
                continue

            for name, pruned in features.items():
                # TODO: add this flags to constants
                old_pruned = m.features_pruned[name]
                feature_pruned = [int(a+b) for a, b in zip(old_pruned, pruned)]
                m.features_pruned[name] = feature_pruned

    def prune_step(self, prune_ratio: float | NAMED_RATIO):
        print(' > Extracting features')
        self.clean_output()
        x, y = next(iter(self.inputs.train))
        train = self.extract_features(x, y)
        x, y = next(iter(self.inputs.test))
        test = self.extract_features(x, y)
        print(' > Evaluating features')
        named_features_importance = self.eval_features(train, test, self.to_prune_modules, prune_ratio)
        prune_features = self._get_prune_features(self.global_pruning, prune_ratio, named_features_importance)
        self.prune_modules(prune_features)

    def adjust_prune_ratio(self, prune_ratio: float | NAMED_RATIO, steps: int):
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


def full_class_name(c) -> str:
    return f'{c.__class__.__module__}.{c.__class__.__name__}'


def binarize(ratio: float, rank: torch.tensor) -> list[int]:
    expected_pruned = int(round(len(rank) * ratio, 1))
    ordered = sorted([(i, x) for i, x in enumerate(rank)], key=lambda x: x[1])
    indexes = [0] * len(rank)

    while sum(indexes) < expected_pruned:
        weakest_idx = ordered.pop(0)[0]
        indexes[weakest_idx] = 1

    return indexes


def store_feature_output(self, x, *args, **kwargs):
    features_output = self.extract_feature_middleware(x, *args, **kwargs)

    temp = tempfile.NamedTemporaryFile(prefix='prune_features_extraction', suffix='.pth')
    torch.save(features_output, temp.name)
    temp.flush()  # Force save to disk
    self.features_output = temp

    return self.original_forward(x, *args, **kwargs)
